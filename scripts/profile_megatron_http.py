#!/usr/bin/env python3
"""Megatron memory profiling via HTTP API.

Uses the correct tinker API:
1. create_model -> get model_id
2. forward_backward with model_id and forward_backward_input

Tests different sequence lengths to profile memory usage.
"""

import argparse
import time
import uuid
import requests
import numpy as np

BASE_URL = "http://localhost:8000"
K2_MODEL = "moonshotai/Kimi-K2-Thinking"


def poll_future(request_id: str, timeout: int = 600) -> dict:
    """Poll for async operation completion."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            json={"request_id": request_id},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(1)
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    return {"error": f"Timeout after {timeout}s"}


def create_model(rank: int = 16) -> str | None:
    """Create a training model, return model_id."""
    session_id = f"profile_{uuid.uuid4().hex[:8]}"
    print(f"Creating model (rank={rank})...")

    resp = requests.post(
        f"{BASE_URL}/api/v1/create_model",
        json={
            "session_id": session_id,
            "model_seq_id": 1,
            "base_model": K2_MODEL,
            "lora_config": {"rank": rank},
            "learning_rate": 1e-4,
        },
        timeout=60,
    )
    if resp.status_code != 200:
        print(f"create_model failed: HTTP {resp.status_code}: {resp.text}")
        return None

    request_id = resp.json().get("request_id")
    print(f"Waiting for model (request_id={request_id})...")

    result = poll_future(request_id, timeout=900)
    if "error" in result:
        print(f"Error: {result['error']}")
        return None

    model_id = result.get("model_id")
    print(f"Model created: {model_id}")
    return model_id


def create_datum(seq_len: int, vocab_size: int = 163840) -> dict:
    """Create a Datum in correct API format."""
    tokens = np.random.randint(0, vocab_size, size=seq_len, dtype=np.int64).tolist()
    target_tokens = tokens[1:] + [tokens[0]]
    loss_mask = [1.0] * seq_len

    return {
        "model_input": {"chunks": [{"tokens": tokens[:-1], "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens[:-1], "shape": [seq_len - 1], "dtype": "int64"},
            "loss_mask": {"data": loss_mask[1:], "shape": [seq_len - 1], "dtype": "float32"},
        },
    }


def run_forward_backward(model_id: str, data: list[dict], timeout: int = 600) -> dict:
    """Run forward_backward via HTTP API."""
    resp = requests.post(
        f"{BASE_URL}/api/v1/forward_backward",
        json={
            "model_id": model_id,
            "forward_backward_input": {
                "data": data,
                "loss_fn": "cross_entropy",
            },
        },
        timeout=30,
    )
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text}"}

    request_id = resp.json().get("request_id")
    return poll_future(request_id, timeout=timeout)


def profile_memory(model_id: str, seq_lengths: list[int], batch_size: int = 1, warmup: int = 1, iterations: int = 2):
    """Profile memory at different sequence lengths."""
    results = []

    for seq_len in seq_lengths:
        print(f"\n{'='*60}")
        print(f"Testing seq_len={seq_len}, batch_size={batch_size}")
        print(f"{'='*60}")

        data = [create_datum(seq_len) for _ in range(batch_size)]
        total_tokens = seq_len * batch_size
        print(f"Total tokens: {total_tokens}")

        # Warmup
        success = True
        for i in range(warmup):
            print(f"Warmup {i+1}/{warmup}...")
            t0 = time.time()
            result = run_forward_backward(model_id, data)
            t1 = time.time()
            if "error" in result:
                error_str = str(result["error"])
                print(f"ERROR: {error_str[:200]}")
                status = "OOM" if "out of memory" in error_str.lower() else "ERROR"
                results.append({"seq_len": seq_len, "status": status, "error": error_str[:200]})
                success = False
                break
            loss = result.get("metrics", {}).get("loss:mean", "N/A")
            print(f"Warmup {i+1}: loss={loss}, time={t1-t0:.2f}s")

        if not success:
            continue

        # Timed iterations
        times = []
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}...")
            t0 = time.time()
            result = run_forward_backward(model_id, data)
            t1 = time.time()
            if "error" in result:
                print(f"ERROR during iteration: {result['error'][:200]}")
                break
            times.append(t1 - t0)
            loss = result.get("metrics", {}).get("loss:mean", "N/A")
            print(f"Iteration {i+1}: loss={loss}, time={t1-t0:.2f}s")

        if times:
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
            print(f"seq_len={seq_len}: avg_time={avg_time:.2f}s, throughput={tokens_per_sec:.0f} tokens/s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Profile K2 Megatron memory via HTTP")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384], help="Sequence lengths")
    parser.add_argument("--batch-size", type=int, default=1, help="Micro batch size")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=2, help="Timed iterations")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    args = parser.parse_args()

    print("=" * 60)
    print("K2 Megatron Memory Profiling (HTTP)")
    print("=" * 60)
    print(f"Model: {K2_MODEL}")
    print(f"LoRA rank: {args.rank}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print()

    model_id = create_model(args.rank)
    if not model_id:
        print("Failed to create model")
        return 1

    results = profile_memory(model_id, args.seq_lengths, args.batch_size, args.warmup, args.iterations)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'seq_len':<10} {'tokens':<10} {'status':<10} {'time(s)':<10} {'tok/s':<10}")
    print("-" * 60)
    for r in results:
        if r["status"] == "OK":
            print(f"{r['seq_len']:<10} {r['total_tokens']:<10} {r['status']:<10} {r['avg_time_s']:<10.2f} {r['tokens_per_sec']:<10.0f}")
        else:
            print(f"{r['seq_len']:<10} {'N/A':<10} {r['status']:<10} {'N/A':<10} {'N/A':<10}")
            if "error" in r:
                print(f"  Error: {r['error']}")

    return 0


if __name__ == "__main__":
    exit(main())
