#!/usr/bin/env python3
"""Phase 11: Concurrent sessions stress test.

Creates 10 concurrent sessions on the same base model and
sends training requests simultaneously. Tests:
- Request queueing behavior
- No race conditions or deadlocks
- Proper session isolation under concurrent load

Run from tinker-server root directory:
    TINKER_BASE_URL=http://localhost:8000 python scripts/test_concurrent_sessions.py
"""

import argparse
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests


def get_base_url():
    return os.environ.get("TINKER_BASE_URL", "http://localhost:8000")


def poll_future(request_id: str, timeout: int = 300) -> dict:
    """Poll for async operation result."""
    poll_url = f"{get_base_url()}/api/v1/retrieve_future"
    start = time.time()

    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(0.5)
            continue
        else:
            resp.raise_for_status()

    raise TimeoutError(f"Operation did not complete within {timeout}s")


def create_session(base_model: str, lora_rank: int = 32) -> tuple[str, str, float]:
    """Create training session and return (session_id, model_id, create_time)."""
    session_id = str(uuid.uuid4())
    url = f"{get_base_url()}/api/v1/create_model"
    payload = {
        "session_id": session_id,
        "model_seq_id": 1,
        "base_model": base_model,
        "lora_config": {"rank": lora_rank},
    }

    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    future = resp.json()
    result = poll_future(future.get("request_id"), timeout=300)
    create_time = time.time() - t0

    if "error" in result:
        raise RuntimeError(f"Session creation failed: {result['error']}")

    return session_id, result.get("model_id"), create_time


def forward_backward(model_id: str, data: list) -> tuple[float, float]:
    """Run one forward-backward pass. Returns (loss, time)."""
    url = f"{get_base_url()}/api/v1/forward_backward"
    payload = {
        "model_id": model_id,
        "forward_backward_input": {
            "data": data,
            "loss_fn": "cross_entropy",
        },
    }

    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    future = resp.json()
    result = poll_future(future.get("request_id"), timeout=300)
    fb_time = time.time() - t0

    if "error" in result:
        raise RuntimeError(f"forward_backward error: {result['error']}")

    loss = result.get("metrics", {}).get("loss:mean", 0)
    if loss == 0:
        loss_outputs = result.get("loss_fn_outputs", [])
        if loss_outputs:
            losses = [o.get("loss", {}).get("data", [0])[0] for o in loss_outputs]
            loss = sum(losses) / len(losses) if losses else 0

    return loss, fb_time


def create_sample_data(session_idx: int, batch_size: int = 2, seq_len: int = 16) -> list:
    """Create minimal sample data with unique tokens per session."""
    data = []
    base_token = 1000 + session_idx * 1000  # Unique tokens per session
    for i in range(batch_size):
        input_tokens = [151644] + list(range(base_token + i * 100, base_token + i * 100 + seq_len))
        target_tokens = list(range(base_token + i * 100, base_token + i * 100 + seq_len)) + [0]
        loss_mask = [1.0] * seq_len + [0.0]

        data.append({
            "model_input": {
                "chunks": [{"tokens": input_tokens, "type": "encoded_text"}]
            },
            "loss_fn_inputs": {
                "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
                "loss_mask": {"data": loss_mask, "shape": [len(loss_mask)], "dtype": "float32"},
            },
        })
    return data


def run_session_task(session_idx: int, model_id: str, num_iters: int) -> dict:
    """Run training iterations for a single session. Called from thread pool."""
    result = {
        "session_idx": session_idx,
        "model_id": model_id,
        "iterations": [],
        "start_time": time.time(),
    }

    try:
        data = create_sample_data(session_idx)
        for i in range(num_iters):
            loss, fb_time = forward_backward(model_id, data)
            result["iterations"].append({
                "iteration": i + 1,
                "loss": loss,
                "fb_time": fb_time,
            })
        result["status"] = "ok"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()

    result["end_time"] = time.time()
    result["total_time"] = result["end_time"] - result["start_time"]
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 11 concurrent sessions stress test")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Model name (default: Qwen3-30B-A3B MoE)",
    )
    parser.add_argument(
        "--num-sessions",
        type=int,
        default=10,
        help="Number of concurrent sessions (default: 10)",
    )
    parser.add_argument(
        "--iters-per-session",
        type=int,
        default=5,
        help="Training iterations per session (default: 5)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank (default: 32)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/concurrent_sessions"),
        help="Output directory for results",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("CONCURRENT SESSIONS STRESS TEST")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Concurrent sessions: {args.num_sessions}")
    print(f"Iterations per session: {args.iters_per_session}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"{'='*60}\n")

    results = {
        "model_name": args.model,
        "num_sessions": args.num_sessions,
        "iters_per_session": args.iters_per_session,
        "lora_rank": args.lora_rank,
        "sessions": [],
        "start_time": datetime.now().isoformat(),
    }

    # Phase 1: Create all sessions first (sequential)
    print("Phase 1: Creating sessions...")
    sessions = []
    for i in range(args.num_sessions):
        try:
            session_id, model_id, create_time = create_session(args.model, args.lora_rank)
            sessions.append({
                "idx": i,
                "session_id": session_id,
                "model_id": model_id,
                "create_time": create_time,
            })
            print(f"  Session {i+1}/{args.num_sessions}: created in {create_time:.2f}s")
        except Exception as e:
            print(f"  Session {i+1}/{args.num_sessions}: FAILED - {e}")
            sessions.append({
                "idx": i,
                "error": str(e),
            })

    results["session_creation"] = sessions
    successful_sessions = [s for s in sessions if "model_id" in s]
    print(f"\nCreated {len(successful_sessions)}/{args.num_sessions} sessions\n")

    if not successful_sessions:
        print("No sessions created, aborting.")
        return

    # Phase 2: Send concurrent training requests
    print("Phase 2: Sending concurrent training requests...")
    concurrent_start = time.time()

    with ThreadPoolExecutor(max_workers=args.num_sessions) as executor:
        futures = {
            executor.submit(
                run_session_task,
                s["idx"],
                s["model_id"],
                args.iters_per_session,
            ): s
            for s in successful_sessions
        }

        for future in as_completed(futures):
            session_info = futures[future]
            try:
                result = future.result()
                results["sessions"].append(result)
                status = "OK" if result["status"] == "ok" else "ERROR"
                print(f"  Session {result['session_idx']+1}: {status} ({result['total_time']:.2f}s)")
            except Exception as e:
                print(f"  Session {session_info['idx']+1}: EXCEPTION - {e}")
                results["sessions"].append({
                    "session_idx": session_info["idx"],
                    "status": "exception",
                    "error": str(e),
                })

    concurrent_time = time.time() - concurrent_start
    results["concurrent_phase_time"] = concurrent_time

    # Calculate statistics
    successful_results = [s for s in results["sessions"] if s.get("status") == "ok"]
    errors = [s for s in results["sessions"] if s.get("status") != "ok"]

    results["num_successful"] = len(successful_results)
    results["num_errors"] = len(errors)
    results["end_time"] = datetime.now().isoformat()

    if successful_results:
        total_iters = sum(len(s.get("iterations", [])) for s in successful_results)
        total_times = [s["total_time"] for s in successful_results]
        results["total_iterations"] = total_iters
        results["avg_session_time"] = sum(total_times) / len(total_times)
        results["min_session_time"] = min(total_times)
        results["max_session_time"] = max(total_times)

        # Throughput: concurrent phase handles all sessions simultaneously
        results["throughput_iters_per_sec"] = total_iters / concurrent_time

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output_dir / f"concurrent_test_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("CONCURRENT SESSIONS TEST COMPLETE")
    print(f"{'='*60}")
    print(f"Sessions successful: {len(successful_results)}/{args.num_sessions}")
    print(f"Concurrent phase time: {concurrent_time:.2f}s")
    if successful_results:
        print(f"Total iterations: {results['total_iterations']}")
        print(f"Throughput: {results['throughput_iters_per_sec']:.2f} iters/sec")
        print(f"Avg session time: {results['avg_session_time']:.2f}s")
        print(f"Min/Max session time: {results['min_session_time']:.2f}s / {results['max_session_time']:.2f}s")
    print(f"Errors: {len(errors)}")
    for s in errors[:5]:
        print(f"  Session {s['session_idx']+1}: {s.get('error', 'unknown')[:100]}")
    print(f"Results saved: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
