#!/usr/bin/env python3
"""Test K2 with TP=8, EP=96, ETP=1, rank=8 for extended context.

Lower rank=8 uses ~50% less LoRA memory than rank=16.
"""
import os
import sys
import time
import uuid
import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
HEADERS = {"Authorization": "Bearer dummy", "Content-Type": "application/json"}


def poll(req_id: str, timeout: int = 1200) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        r = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            json={"request_id": req_id},
            headers=HEADERS,
            timeout=120,
        )
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 408:
            elapsed = int(time.time() - start)
            print(f"    {elapsed}s...", flush=True)
            time.sleep(15)
        else:
            return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
    return {"error": "timeout"}


def main():
    print("=" * 60)
    print("Testing K2: TP=8, EP=96, ETP=1, rank=8 for extended context")
    print("=" * 60)
    print("Config: 96 GPUs, rank=8 (lower memory)")
    print()

    # Kill existing actor
    print("Killing existing Megatron actor...")
    r = requests.delete(
        f"{BASE_URL}/api/v1/megatron/moonshotai%2FKimi-K2-Thinking",
        headers=HEADERS,
        timeout=30,
    )
    print(f"  Status: {r.status_code}")
    time.sleep(5)

    # Create model with rank=8
    sid = f"k2_tp8_rank8_{uuid.uuid4().hex[:8]}"
    print(f"\nCreating model with rank=8...")
    r = requests.post(
        f"{BASE_URL}/api/v1/create_model",
        json={
            "session_id": sid,
            "model_seq_id": 1,
            "base_model": "moonshotai/Kimi-K2-Thinking",
            "lora_config": {"rank": 8},
            "learning_rate": 1e-5,
        },
        headers=HEADERS,
        timeout=60,
    )
    if r.status_code != 200:
        print(f"FAIL create_model: {r.status_code}: {r.text[:500]}")
        sys.exit(1)

    result = poll(r.json().get("request_id"), timeout=1200)
    if "error" in result:
        print(f"FAIL model init: {result['error']}")
        sys.exit(1)

    model_id = result.get("model_id")
    print(f"Model ready: {model_id}")

    # Test context lengths - focus on 15K-30K range
    test_lengths = [15360, 20480, 25600, 30720]
    max_working = 0

    for seq_len in test_lengths:
        print(f"\nTesting {seq_len} tokens...", end="", flush=True)
        tokens = [100] * seq_len
        payload = {
            "model_id": model_id,
            "forward_backward_input": {
                "data": [
                    {
                        "model_input": {
                            "chunks": [{"tokens": tokens[:-1], "type": "encoded_text"}]
                        },
                        "loss_fn_inputs": {
                            "target_tokens": {
                                "data": tokens[1:],
                                "shape": [seq_len - 1],
                                "dtype": "int64",
                            },
                            "loss_mask": {
                                "data": [1.0] * (seq_len - 1),
                                "shape": [seq_len - 1],
                                "dtype": "float32",
                            },
                        },
                    }
                ],
                "loss_fn": "cross_entropy",
            },
            "session_id": sid,
        }
        start = time.time()
        r = requests.post(
            f"{BASE_URL}/api/v1/forward_backward",
            json=payload,
            headers=HEADERS,
            timeout=60,
        )
        if r.status_code != 200:
            print(f" HTTP ERROR: {r.status_code}")
            break

        result = poll(r.json().get("request_id"), timeout=600)
        elapsed = time.time() - start
        if "error" in result:
            err = str(result["error"])
            if (
                "out of memory" in err.lower()
                or "oom" in err.lower()
                or "cuda" in err.lower()
            ):
                print(f" OOM ({elapsed:.1f}s)")
            else:
                print(f" FAIL ({elapsed:.1f}s)")
                print(f"    Error: {err[:200]}")
            break
        else:
            loss = result.get("metrics", {}).get("loss:mean", result.get("loss", "N/A"))
            print(f" PASS: loss={loss:.4f} ({elapsed:.1f}s)")
            max_working = seq_len

    print("\n" + "=" * 60)
    print(f"Max working context with TP=8, EP=96, rank=8: {max_working}")
    print("=" * 60)


if __name__ == "__main__":
    main()
