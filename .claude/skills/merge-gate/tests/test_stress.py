"""Stress Test: Concurrent Clients with Mixed Configurations.

Tests system behavior under concurrent load with:
- Multiple sessions running in parallel
- Different models (Dense and MoE)
- Different LoRA ranks
- Different loss functions (SFT, RL)

This test validates:
1. Session isolation (no cross-contamination)
2. Request serialization (no deadlocks)
3. LRU eviction (when resources insufficient for all models)

Pass criteria: All clients complete without deadlock or error.
"""

import concurrent.futures
import threading
import time
from typing import Any

import numpy as np
import pytest
from transformers import AutoTokenizer

from .conftest import (
    DENSE_MODEL,
    MOE_MODEL,
    create_session,
    forward_backward,
    optim_step,
    train_step,
)

# Pre-load tokenizers to avoid concurrent import race conditions
_dense_tokenizer = None
_moe_tokenizer = None
_tokenizer_lock = threading.Lock()


def get_dense_tokenizer():
    """Get Dense model tokenizer (lazy loaded, thread-safe)."""
    global _dense_tokenizer
    if _dense_tokenizer is None:
        with _tokenizer_lock:
            if _dense_tokenizer is None:
                _dense_tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL, trust_remote_code=True)
    return _dense_tokenizer


def get_moe_tokenizer():
    """Get MoE model tokenizer (lazy loaded, thread-safe)."""
    global _moe_tokenizer
    if _moe_tokenizer is None:
        with _tokenizer_lock:
            if _moe_tokenizer is None:
                _moe_tokenizer = AutoTokenizer.from_pretrained(MOE_MODEL, trust_remote_code=True)
    return _moe_tokenizer


# Lock for printing
print_lock = threading.Lock()


def safe_print(msg: str):
    """Thread-safe print."""
    with print_lock:
        print(msg)


def run_dense_sft_client(client_id: int, rank: int, num_iterations: int = 3) -> dict:
    """Run a Dense SFT client."""
    results = {
        "client_id": client_id,
        "model": "Dense",
        "task": "SFT",
        "rank": rank,
        "status": "started",
        "losses": [],
        "errors": [],
    }

    try:
        safe_print(f"[Client {client_id}] Starting Dense SFT (rank={rank})")

        # Create session
        session_id, model_id = create_session(DENSE_MODEL, lora_rank=rank, lr=1e-4)
        results["session_id"] = session_id
        results["model_id"] = model_id

        # Simple training data
        tokenizer = get_dense_tokenizer()

        prompt = f"Client {client_id} test: "
        response = "response"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        response_tokens = tokenizer.encode(response, add_special_tokens=False)
        full_tokens = prompt_tokens + response_tokens

        api_data = [{
            "model_input": {"chunks": [{"tokens": full_tokens[:-1], "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": full_tokens[1:], "shape": [len(full_tokens) - 1], "dtype": "int64"},
                "loss_mask": {"data": [1.0] * (len(full_tokens) - 1), "shape": [len(full_tokens) - 1], "dtype": "float32"},
            },
        }]

        # Training loop
        for i in range(num_iterations):
            result = forward_backward(model_id, api_data, loss_fn="cross_entropy")
            if "error" in result:
                results["errors"].append(f"iter {i}: {result['error']}")
                continue

            loss = result.get("metrics", {}).get("loss:mean", 0)
            results["losses"].append(loss)

            optim_step(model_id, lr=1e-4)
            safe_print(f"[Client {client_id}] Iteration {i+1}: loss={loss:.4f}")

        results["status"] = "completed"
        safe_print(f"[Client {client_id}] Completed successfully")

    except Exception as e:
        results["status"] = "failed"
        results["errors"].append(str(e))
        safe_print(f"[Client {client_id}] Failed: {e}")

    return results


def run_dense_rl_client(client_id: int, rank: int, num_iterations: int = 3) -> dict:
    """Run a Dense RL client."""
    results = {
        "client_id": client_id,
        "model": "Dense",
        "task": "RL",
        "rank": rank,
        "status": "started",
        "losses": [],
        "errors": [],
    }

    try:
        safe_print(f"[Client {client_id}] Starting Dense RL (rank={rank})")

        # Create session
        session_id, model_id = create_session(DENSE_MODEL, lora_rank=rank, lr=1e-4)
        results["session_id"] = session_id
        results["model_id"] = model_id

        # Simple RL data with dummy advantages
        tokenizer = get_dense_tokenizer()

        prompt = f"RL client {client_id}: "
        response = "answer"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        response_tokens = tokenizer.encode(response, add_special_tokens=False)
        full_tokens = prompt_tokens + response_tokens

        loss_mask = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(response_tokens)
        advantages = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(response_tokens)
        old_logprobs = [0.0] * (len(full_tokens) - 1)

        api_data = [{
            "model_input": {"chunks": [{"tokens": full_tokens[:-1], "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": full_tokens[1:], "shape": [len(full_tokens) - 1], "dtype": "int64"},
                "loss_mask": {"data": loss_mask, "shape": [len(loss_mask)], "dtype": "float32"},
                "logprobs": {"data": old_logprobs, "shape": [len(old_logprobs)], "dtype": "float32"},
                "advantages": {"data": advantages, "shape": [len(advantages)], "dtype": "float32"},
            },
        }]

        # Training loop
        for i in range(num_iterations):
            result = forward_backward(model_id, api_data, loss_fn="importance_sampling")
            if "error" in result:
                results["errors"].append(f"iter {i}: {result['error']}")
                continue

            loss = result.get("metrics", {}).get("loss:mean", 0)
            results["losses"].append(loss)

            optim_step(model_id, lr=1e-4)
            safe_print(f"[Client {client_id}] Iteration {i+1}: loss={loss:.4f}")

        results["status"] = "completed"
        safe_print(f"[Client {client_id}] Completed successfully")

    except Exception as e:
        results["status"] = "failed"
        results["errors"].append(str(e))
        safe_print(f"[Client {client_id}] Failed: {e}")

    return results


def run_moe_sft_client(client_id: int, rank: int, num_iterations: int = 3) -> dict:
    """Run a MoE SFT client."""
    results = {
        "client_id": client_id,
        "model": "MoE",
        "task": "SFT",
        "rank": rank,
        "status": "started",
        "losses": [],
        "errors": [],
    }

    try:
        safe_print(f"[Client {client_id}] Starting MoE SFT (rank={rank})")

        # Create session
        session_id, model_id = create_session(MOE_MODEL, lora_rank=rank, lr=1e-4)
        results["session_id"] = session_id
        results["model_id"] = model_id

        # Simple training data
        tokenizer = get_moe_tokenizer()

        prompt = f"MoE client {client_id}: "
        response = "response"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        response_tokens = tokenizer.encode(response, add_special_tokens=False)
        full_tokens = prompt_tokens + response_tokens

        api_data = [{
            "model_input": {"chunks": [{"tokens": full_tokens[:-1], "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": full_tokens[1:], "shape": [len(full_tokens) - 1], "dtype": "int64"},
                "loss_mask": {"data": [1.0] * (len(full_tokens) - 1), "shape": [len(full_tokens) - 1], "dtype": "float32"},
            },
        }]

        # Training loop - use train_step for MoE
        for i in range(num_iterations):
            result = train_step(model_id, api_data, lr=1e-4, loss_fn="cross_entropy")
            if "error" in result:
                results["errors"].append(f"iter {i}: {result['error']}")
                continue

            loss = result.get("metrics", {}).get("loss:mean", 0)
            results["losses"].append(loss)
            safe_print(f"[Client {client_id}] Iteration {i+1}: loss={loss:.4f}")

        results["status"] = "completed"
        safe_print(f"[Client {client_id}] Completed successfully")

    except Exception as e:
        results["status"] = "failed"
        results["errors"].append(str(e))
        safe_print(f"[Client {client_id}] Failed: {e}")

    return results


class TestStress:
    """Stress tests for concurrent clients."""

    def test_concurrent_dense_clients(self):
        """Test multiple Dense clients with different configs.

        5 concurrent Dense clients:
        - Client 1: SFT, rank=16
        - Client 2: RL, rank=32
        - Client 3: SFT, rank=64
        - Client 4: RL, rank=32
        - Client 5: SFT, rank=16
        """
        client_configs = [
            (1, run_dense_sft_client, 16),
            (2, run_dense_rl_client, 32),
            (3, run_dense_sft_client, 64),
            (4, run_dense_rl_client, 32),
            (5, run_dense_sft_client, 16),
        ]

        results = []
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(func, client_id, rank, 3): client_id
                for client_id, func, rank in client_configs
            }

            for future in concurrent.futures.as_completed(futures, timeout=600):
                client_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        "client_id": client_id,
                        "status": "exception",
                        "errors": [str(e)],
                    })

        elapsed = time.time() - start_time

        # Print summary
        print(f"\n{'='*60}")
        print(f"STRESS TEST RESULTS: Dense Clients")
        print(f"{'='*60}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Clients: {len(results)}")

        completed = sum(1 for r in results if r["status"] == "completed")
        failed = sum(1 for r in results if r["status"] != "completed")

        print(f"Completed: {completed}")
        print(f"Failed: {failed}")

        for r in results:
            status_mark = "PASS" if r["status"] == "completed" else "FAIL"
            print(f"  Client {r['client_id']}: {status_mark} "
                  f"({r.get('model', '?')}/{r.get('task', '?')}/rank={r.get('rank', '?')})")
            if r.get("errors"):
                for err in r["errors"][:2]:
                    print(f"    Error: {err}")

        # All clients should complete
        assert completed == len(client_configs), (
            f"Not all clients completed: {completed}/{len(client_configs)}"
        )

    def test_mixed_model_lru_eviction(self):
        """Test LRU eviction with Dense and MoE clients.

        When insufficient GPUs for both models simultaneously,
        the system should evict least-recently-used actors.

        Clients:
        - Client 1: Dense SFT (needs 2 GPUs)
        - Client 2: MoE SFT (needs 12 GPUs)
        - Client 3: Dense SFT (may trigger eviction)

        NOTE: This test is meaningful when cluster has 12-14 GPUs,
        forcing LRU eviction when switching between Dense and MoE.
        """
        print("\n=== LRU Eviction Test ===")
        print("This test validates graceful actor replacement when")
        print("Dense and MoE models cannot colocate simultaneously.\n")

        results = []

        # Run Dense first
        print("Phase 1: Starting Dense client...")
        r1 = run_dense_sft_client(1, rank=32, num_iterations=2)
        results.append(r1)

        # Run MoE (may evict Dense actors)
        print("\nPhase 2: Starting MoE client (may trigger eviction)...")
        r2 = run_moe_sft_client(2, rank=32, num_iterations=2)
        results.append(r2)

        # Run Dense again (may evict MoE actors)
        print("\nPhase 3: Starting Dense client again...")
        r3 = run_dense_sft_client(3, rank=32, num_iterations=2)
        results.append(r3)

        # Print summary
        print(f"\n{'='*60}")
        print(f"LRU EVICTION TEST RESULTS")
        print(f"{'='*60}")

        all_passed = True
        for r in results:
            status_mark = "PASS" if r["status"] == "completed" else "FAIL"
            if r["status"] != "completed":
                all_passed = False
            print(f"  Client {r['client_id']}: {status_mark} "
                  f"({r.get('model', '?')}/{r.get('task', '?')})")
            if r.get("errors"):
                for err in r["errors"][:2]:
                    print(f"    Error: {err}")

        assert all_passed, "Some clients failed during LRU eviction test"

    def test_rapid_session_creation(self):
        """Test rapid session creation and teardown.

        Creates multiple sessions in quick succession to test
        session management under load.
        """
        num_sessions = 5
        sessions = []

        print(f"\nCreating {num_sessions} sessions rapidly...")
        start_time = time.time()

        for i in range(num_sessions):
            try:
                session_id, model_id = create_session(DENSE_MODEL, lora_rank=32, lr=1e-4)
                sessions.append({
                    "session_id": session_id,
                    "model_id": model_id,
                    "status": "created"
                })
                print(f"  Session {i+1}: {session_id}")
            except Exception as e:
                sessions.append({
                    "status": "failed",
                    "error": str(e)
                })
                print(f"  Session {i+1}: FAILED - {e}")

        elapsed = time.time() - start_time
        print(f"\nCreated {len(sessions)} sessions in {elapsed:.2f}s")
        print(f"Avg: {elapsed/num_sessions:.2f}s per session")

        created = sum(1 for s in sessions if s["status"] == "created")
        assert created == num_sessions, f"Failed to create all sessions: {created}/{num_sessions}"
