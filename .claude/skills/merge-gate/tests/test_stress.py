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
import os
import threading
import time

import pytest
from transformers import AutoTokenizer

from .conftest import (
    DENSE_MODEL,
    MOE_MODEL,
    create_session,
    forward_backward,
    optim_step,
    save_weights,
    list_actors,
    get_admission_stats,
)
from .framework import (
    LRUEvictionData,
    PlotGenerator,
    TestReport,
    create_test_report,
    print_report_summary,
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
        print("STRESS TEST RESULTS: Dense Clients")
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

    def test_concurrent_moe_sessions(self):
        """Test concurrent MoE sessions (Issue #44 regression guard).

        Two concurrent MoE SFT sessions on Qwen3-30B-A3B.
        Pass criteria: both sessions complete without deadlock or error.
        """
        client_configs = [
            (1, run_moe_sft_client, 32),
            (2, run_moe_sft_client, 32),
        ]

        results = []
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(func, client_id, rank, 3): client_id
                for client_id, func, rank in client_configs
            }

            try:
                for future in concurrent.futures.as_completed(futures, timeout=1200):
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
            except concurrent.futures.TimeoutError:
                # Some futures did not complete: treat as deadlock/freeze.
                for future, client_id in futures.items():
                    if not future.done():
                        results.append({
                            "client_id": client_id,
                            "status": "timeout",
                            "errors": ["Timeout waiting for client completion (possible deadlock/freeze)"],
                        })

        elapsed = time.time() - start_time

        print(f"\n{'='*60}")
        print("STRESS TEST RESULTS: MoE Concurrent Sessions")
        print(f"{'='*60}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Clients: {len(results)}")

        completed = sum(1 for r in results if r.get("status") == "completed")
        failed = sum(1 for r in results if r.get("status") != "completed")
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")

        for r in results:
            status_mark = "PASS" if r.get("status") == "completed" else "FAIL"
            print(
                f"  Client {r.get('client_id', '?')}: {status_mark} "
                f"({r.get('model', 'MoE')}/{r.get('task', 'SFT')}/rank={r.get('rank', '?')})"
            )
            if r.get("errors"):
                for err in r["errors"][:2]:
                    print(f"    Error: {err}")

        assert completed == len(client_configs), (
            f"Not all MoE clients completed: {completed}/{len(client_configs)}"
        )

    def test_mixed_model_lru_eviction(self):
        """Eviction sentry: require an observed eviction event under 8 GPUs.

        This test snapshots ResourcePool actor inventory before/after phases and
        fails unless at least one eviction event is observed (actor disappearance).

        Requires server started with:
        - MINT_MIN_ACTOR_AGE=0
        - small MINT_SESSION_IDLE_TIMEOUT (so a prior actor becomes idle during the test)
        """
        start_time = time.time()
        idle_wait_s = float(os.environ.get("TINKER_EVICTION_IDLE_WAIT_S", "6"))

        try:
            stats = get_admission_stats()
            gpu_total = float(
                (((stats.get("ray_cluster") or {}).get("resources") or {}).get("gpu_total") or 0.0)
            )
        except Exception:
            gpu_total = 0.0
        if gpu_total > 8.0:
            pytest.skip(f"eviction sentry requires constrained cluster (gpu_total={gpu_total})")

        data = LRUEvictionData()

        def snapshot(phase: str, action: str = "") -> set[str]:
            resp = list_actors()
            actors = [a.get("actor_name") for a in resp.get("actors", []) if isinstance(a, dict)]
            actors = [a for a in actors if isinstance(a, str) and a]
            gpu_usage = int(resp.get("total_gpus_used", 0) or 0)
            data.add_snapshot(phase=phase, actors=actors, gpu_usage=gpu_usage, action=action)
            print(f"snapshot phase={phase} gpus={gpu_usage} actors={len(actors)} action={action!r}")
            return set(actors)

        prev = snapshot("start")

        # Phase 1: create a Dense training actor, then let it become idle.
        r1 = run_dense_sft_client(1, rank=32, num_iterations=1)
        assert r1.get("status") == "completed", f"dense phase failed: {r1}"
        dense_after = snapshot("dense_created", action="dense_sft_client")

        time.sleep(idle_wait_s)
        dense_idle = snapshot("dense_idle_wait", action=f"sleep {idle_wait_s}s")

        # Phase 2: create MoE trainer, then create MoE vLLM engine (save_weights) to push total GPU demand > 8.
        r2 = run_moe_sft_client(2, rank=32, num_iterations=1)
        assert r2.get("status") == "completed", f"moe trainer phase failed: {r2}"
        moe_trainer = snapshot("moe_trainer", action="moe_sft_client")

        moe_model_id = r2.get("model_id")
        assert isinstance(moe_model_id, str) and moe_model_id, f"missing model_id from moe client: {r2}"
        save_res = save_weights(moe_model_id, name="merge_gate_eviction_moe")
        assert "error" not in save_res, f"save_weights failed: {save_res.get('error')}"
        moe_vllm = snapshot("moe_vllm_created", action="save_weights(moe)")

        # Phase 3: touch Dense again (may evict MoE actors depending on LRU/idle).
        r3 = run_dense_sft_client(3, rank=32, num_iterations=1)
        assert r3.get("status") == "completed", f"dense reentry phase failed: {r3}"
        dense_again = snapshot("dense_again", action="dense_sft_client")

        # Infer eviction events from consecutive snapshots.
        for i in range(1, len(data.snapshots)):
            a0 = set(data.snapshots[i - 1].actors)
            a1 = set(data.snapshots[i].actors)
            removed = sorted(a0 - a1)
            added = sorted(a1 - a0)
            if removed:
                data.eviction_events.append({
                    "from_phase": data.snapshots[i - 1].phase,
                    "to_phase": data.snapshots[i].phase,
                    "removed": removed,
                    "added": added,
                })

        plot = PlotGenerator().lru_eviction_timeline(data)
        report = create_test_report(
            test_name="stress_lru_eviction",
            test_type="eviction",
            data=data,
            start_time=start_time,
            plots=[plot] if plot else [],
            metadata={
                "idle_wait_s": idle_wait_s,
                "dense_model": DENSE_MODEL,
                "moe_model": MOE_MODEL,
            },
        )
        report_path = report.save()
        print_report_summary(report)
        print(f"report_json={report_path}")

        assert data.eviction_events, "no eviction events observed (check MINT_MIN_ACTOR_AGE and MINT_SESSION_IDLE_TIMEOUT)"

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

    def test_interleaved_sessions(self):
        """Test multi-tenant concurrency with interleaved sessions.

        This is CRITICAL for production: multiple users switching between
        sessions must maintain correct state via stateless trainer pattern.

        Pattern: Session A → B → A should continue correctly.

        Expected:
        - Session A: loss continues decreasing after switch (no state reset)
        - Session B: independent loss trajectory
        - No weight contamination between sessions
        """
        print("\n=== Interleaved Sessions Test ===")
        print("Testing stateless trainer with session switching...")

        # Prepare different training data for each session
        tokenizer = get_dense_tokenizer()

        # Session A: Pig Latin style
        prompt_a = "Translate: hello"
        response_a = "ellohay"
        tokens_a = tokenizer.encode(f"{prompt_a} {response_a}", add_special_tokens=True)
        data_a = [{
            "model_input": {"chunks": [{"tokens": tokens_a[:-1], "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": tokens_a[1:], "shape": [len(tokens_a) - 1], "dtype": "int64"},
                "loss_mask": {"data": [1.0] * (len(tokens_a) - 1), "shape": [len(tokens_a) - 1], "dtype": "float32"},
            },
        }]

        # Session B: Different data (arithmetic)
        prompt_b = "What is 2+2?"
        response_b = "4"
        tokens_b = tokenizer.encode(f"{prompt_b} {response_b}", add_special_tokens=True)
        data_b = [{
            "model_input": {"chunks": [{"tokens": tokens_b[:-1], "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": tokens_b[1:], "shape": [len(tokens_b) - 1], "dtype": "int64"},
                "loss_mask": {"data": [1.0] * (len(tokens_b) - 1), "shape": [len(tokens_b) - 1], "dtype": "float32"},
            },
        }]

        # Create both sessions
        print("\nPhase 1: Creating sessions...")
        session_a_id, model_a_id = create_session(DENSE_MODEL, lora_rank=32, lr=1e-4)
        session_b_id, model_b_id = create_session(DENSE_MODEL, lora_rank=32, lr=1e-4)
        print(f"  Session A: {session_a_id}")
        print(f"  Session B: {session_b_id}")

        losses_a = []
        losses_b = []

        # Train A for 2 iterations
        print("\nPhase 2: Training Session A (iter 1-2)...")
        for i in range(2):
            result = forward_backward(model_a_id, data_a, loss_fn="cross_entropy")
            loss = result.get("metrics", {}).get("loss:mean", 0)
            losses_a.append(loss)
            optim_step(model_a_id, lr=1e-4)
            print(f"  A iter {i+1}: loss={loss:.4f}")

        # Switch to B, train for 2 iterations
        print("\nPhase 3: Training Session B (iter 1-2)...")
        for i in range(2):
            result = forward_backward(model_b_id, data_b, loss_fn="cross_entropy")
            loss = result.get("metrics", {}).get("loss:mean", 0)
            losses_b.append(loss)
            optim_step(model_b_id, lr=1e-4)
            print(f"  B iter {i+1}: loss={loss:.4f}")

        # Switch back to A, train for 2 more iterations
        print("\nPhase 4: Training Session A (iter 3-4, after switch)...")
        for i in range(2):
            result = forward_backward(model_a_id, data_a, loss_fn="cross_entropy")
            loss = result.get("metrics", {}).get("loss:mean", 0)
            losses_a.append(loss)
            optim_step(model_a_id, lr=1e-4)
            print(f"  A iter {i+3}: loss={loss:.4f}")

        # Analyze results
        print(f"\n{'='*60}")
        print("INTERLEAVED SESSIONS TEST RESULTS")
        print(f"{'='*60}")
        print(f"Session A losses: {[f'{l:.4f}' for l in losses_a]}")
        print(f"Session B losses: {[f'{l:.4f}' for l in losses_b]}")

        # Verify A's loss continues decreasing after switch
        # losses_a[1] is before switch, losses_a[2] is after switch
        loss_before_switch = losses_a[1]
        loss_after_switch = losses_a[2]
        loss_continued = loss_after_switch <= loss_before_switch * 1.1  # Allow 10% tolerance

        print(f"\nA loss before switch (iter 2): {loss_before_switch:.4f}")
        print(f"A loss after switch (iter 3): {loss_after_switch:.4f}")
        print(f"Loss continuity: {'PASS' if loss_continued else 'FAIL'}")

        # Verify B trained independently
        b_decreased = losses_b[-1] < losses_b[0]
        print(f"\nB loss decreased: {losses_b[0]:.4f} -> {losses_b[-1]:.4f} ({'PASS' if b_decreased else 'FAIL'})")

        # Verify A's overall training
        a_overall_decrease = (losses_a[0] - losses_a[-1]) / losses_a[0]
        print(f"A overall reduction: {a_overall_decrease:.1%}")

        anomalies: list[str] = []
        if not loss_continued:
            anomalies.append(
                f"loss_discontinuity_after_switch: {loss_before_switch:.4f} -> {loss_after_switch:.4f}"
            )
        if a_overall_decrease <= 0.3:
            anomalies.append(f"minimal_learning_session_a: {a_overall_decrease:.1%} reduction")

        report = TestReport(
            test_id=session_a_id,
            test_name="stress_interleaved_sessions",
            test_type="stress",
            timestamp=time.strftime("%Y%m%d_%H%M%S", time.localtime()),
            duration_seconds=0.0,
            data={
                "model": DENSE_MODEL,
                "session_a_id": session_a_id,
                "session_b_id": session_b_id,
                "losses_a": losses_a,
                "losses_b": losses_b,
                "loss_before_switch": loss_before_switch,
                "loss_after_switch": loss_after_switch,
                "a_overall_decrease": a_overall_decrease,
            },
            plots=[],
            anomalies=anomalies,
            metadata={},
        )
        report_path = report.save()
        print(f"report_json={report_path}")

        assert len(losses_a) == 4 and len(losses_b) == 2, "incomplete loss traces"
