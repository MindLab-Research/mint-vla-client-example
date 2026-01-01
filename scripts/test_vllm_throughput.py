#!/usr/bin/env python3
"""Test vLLM throughput for K2-Thinking.

Investigates low generation throughput (observed: ~3.81 tok/s).
"""

import os
import sys
import time
import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
MODEL = "moonshotai/Kimi-K2-Thinking"


def poll_future(request_id: str, timeout: int = 600):
    """Poll for async operation completion."""
    poll_url = f"{BASE_URL}/api/v1/retrieve_future"
    start = time.time()
    last_print = 0
    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id}, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            elapsed = time.time() - start
            if elapsed - last_print > 30:
                print(f"    Waiting... {elapsed:.0f}s")
                last_print = elapsed
            time.sleep(0.5)
            continue
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    return {"error": f"Timeout after {timeout}s"}


def get_session():
    """Get existing session or create new one."""
    # Check for existing session by trying a simple request
    print(f"Connecting to existing vLLM session for {MODEL}...")

    # Try to create a session (will be fast if vLLM already loaded)
    resp = requests.post(
        f"{BASE_URL}/api/v1/create_sampling_session",
        json={"base_model": MODEL, "session_id": f"throughput_test_{int(time.time())}"},
        timeout=900,
    )
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}: {resp.text}"

    data = resp.json()
    if "request_id" in data:
        result = poll_future(data["request_id"], timeout=900)
        if "error" in result:
            return None, result["error"]
        return result.get("sampling_session_id"), None
    else:
        return data.get("sampling_session_id"), None


def test_throughput(session_id: str, prompt_len: int, gen_tokens: int, temperature: float = 0.0):
    """Test throughput with specific parameters."""
    prompt_tokens = [100] * prompt_len

    payload = {
        "prompt": {
            "chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]
        },
        "sampling_session_id": session_id,
        "sampling_params": {
            "max_tokens": gen_tokens,
            "temperature": temperature,
        },
        "num_samples": 1,
    }

    start = time.time()
    resp = requests.post(f"{BASE_URL}/api/v1/asample", json=payload, timeout=60)
    submit_time = time.time() - start

    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text}"}

    request_id = resp.json().get("request_id")
    if not request_id:
        return {"error": "No request_id"}

    result = poll_future(request_id, timeout=300)
    total_time = time.time() - start

    if "error" in result:
        return {"error": result["error"], "time": total_time}

    sequences = result.get("sequences", [])
    if not sequences:
        return {"error": "No sequences", "time": total_time}

    seq = sequences[0]
    output_tokens = len(seq.get("tokens", []))
    stop_reason = seq.get("stop_reason", "unknown")

    # Calculate throughput
    decode_time = total_time - submit_time
    tok_per_sec = output_tokens / decode_time if decode_time > 0 else 0

    return {
        "prompt_tokens": prompt_len,
        "output_tokens": output_tokens,
        "stop_reason": stop_reason,
        "submit_time": submit_time,
        "total_time": total_time,
        "decode_time": decode_time,
        "tok_per_sec": tok_per_sec,
    }


def main():
    print("=" * 70)
    print("vLLM Throughput Investigation")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print()

    session_id, error = get_session()
    if error:
        print(f"Error: {error}")
        return 1

    print(f"Session ID: {session_id}")

    # Test matrix: different prompt lengths and generation lengths
    test_cases = [
        # (prompt_len, gen_tokens, description)
        (128, 50, "Short prompt, medium gen"),
        (128, 100, "Short prompt, long gen"),
        (128, 200, "Short prompt, very long gen"),
        (1024, 50, "Medium prompt, medium gen"),
        (1024, 100, "Medium prompt, long gen"),
        (4096, 50, "Long prompt, medium gen"),
        (8192, 50, "Very long prompt, medium gen"),
    ]

    print("\n" + "-" * 70)
    print("Throughput Tests (temperature=0.0 for determinism)")
    print("-" * 70)
    print(f"{'Test':<35} | {'Out':>5} | {'Time':>7} | {'Tok/s':>8} | Stop")
    print("-" * 70)

    results = []
    for prompt_len, gen_tokens, desc in test_cases:
        print(f"  Testing: {desc}...", end="", flush=True)
        result = test_throughput(session_id, prompt_len, gen_tokens)

        if "error" in result:
            print(f" FAIL: {result['error'][:50]}")
        else:
            results.append(result)
            print(f"\r{desc:<35} | {result['output_tokens']:>5} | {result['total_time']:>6.1f}s | {result['tok_per_sec']:>7.2f} | {result['stop_reason']}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)

    if results:
        avg_tok_per_sec = sum(r["tok_per_sec"] for r in results) / len(results)
        max_tok_per_sec = max(r["tok_per_sec"] for r in results)

        print(f"\nAverage throughput: {avg_tok_per_sec:.2f} tok/s")
        print(f"Best throughput:    {max_tok_per_sec:.2f} tok/s")

        # Check if early stopping is occurring
        early_stops = [r for r in results if r["stop_reason"] not in ("length", "stop")]
        if early_stops:
            print(f"\nWARNING: {len(early_stops)} tests stopped early:")
            for r in early_stops:
                print(f"  - {r['prompt_tokens']} prompt -> {r['output_tokens']} output ({r['stop_reason']})")

        # Analyze prefill vs decode
        print("\nTime breakdown:")
        for r in results:
            prefill_pct = (r["submit_time"] / r["total_time"]) * 100 if r["total_time"] > 0 else 0
            decode_pct = 100 - prefill_pct
            print(f"  {r['prompt_tokens']:>5} prompt: submit {r['submit_time']:.2f}s ({prefill_pct:.0f}%), "
                  f"decode {r['decode_time']:.2f}s ({decode_pct:.0f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
