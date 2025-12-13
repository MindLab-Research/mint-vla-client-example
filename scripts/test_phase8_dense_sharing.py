#!/usr/bin/env python3
"""Phase 8 Test: Dense Model Multi-Session Sharing.

Tests:
1. Create multiple sessions with same dense base model
2. Verify DenseTrainerPool reuses actors (fast create times after first)
3. Train sessions with different data (session isolation)
4. Test different LoRA ranks via unified rank support
5. Compare session create times with Phase 6 MoE results

Run from tinker-server root:
    TINKER_BASE_URL=http://localhost:8000 python scripts/test_phase8_dense_sharing.py
"""

import argparse
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests
import matplotlib.pyplot as plt
import numpy as np


def get_base_url():
    return os.environ.get("TINKER_BASE_URL", "http://localhost:8000")


def poll_future(request_id: str, timeout: int = 300) -> dict:
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


def create_session(base_model: str, lora_rank: int, session_name: str) -> tuple[str, str, float, str]:
    """Create training session. Returns (session_id, model_id, create_time, backend)."""
    session_id = f"phase8_{session_name}_{uuid.uuid4().hex[:8]}"
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
    result = poll_future(resp.json().get("request_id"), timeout=300)
    create_time = time.time() - t0

    if "error" in result:
        raise RuntimeError(f"Session creation failed: {result['error']}")

    return session_id, result.get("model_id"), create_time, result.get("backend", "unknown")


def forward_backward(model_id: str, token_base: int = 1000) -> tuple[float, float]:
    data = []
    for i in range(2):
        tokens = [151644] + list(range(token_base + i * 100, token_base + i * 100 + 32))
        targets = list(range(token_base + i * 100, token_base + i * 100 + 32)) + [0]
        data.append({
            "model_input": {"chunks": [{"tokens": tokens, "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": targets, "shape": [len(targets)], "dtype": "int64"},
                "mask": {"data": [1.0] * 32 + [0.0], "shape": [33], "dtype": "float32"},
            },
        })

    url = f"{get_base_url()}/api/v1/forward_backward"
    payload = {"model_id": model_id, "forward_backward_input": {"data": data, "loss_fn": "cross_entropy"}}
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=300)
    if "error" in result:
        raise RuntimeError(f"forward_backward error: {result['error']}")
    loss = result.get("metrics", {}).get("loss:mean", 0)
    if loss == 0:
        loss_outputs = result.get("loss_fn_outputs", [])
        if loss_outputs:
            losses = [o.get("loss", {}).get("data", [0])[0] for o in loss_outputs]
            loss = sum(losses) / len(losses) if losses else 0
    return loss, time.time() - t0


def optim_step(model_id: str) -> float:
    url = f"{get_base_url()}/api/v1/optim_step"
    payload = {
        "model_id": model_id,
        "adam_params": {"learning_rate": 1e-4, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12},
    }
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    poll_future(resp.json().get("request_id"), timeout=300)
    return time.time() - t0


def get_dense_pool_status() -> dict:
    """Get DenseTrainerPool status."""
    url = f"{get_base_url()}/api/v1/dense_pool_status"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def plot_results(results: dict, output_path: Path):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sessions = results["sessions"]
    session_names = list(sessions.keys())

    # Plot 1: Loss vs Iteration for all sessions
    colors = ["b", "r", "g", "orange", "purple"]
    for idx, name in enumerate(session_names):
        data = sessions[name]
        iters = [r["iteration"] for r in data["iterations"]]
        losses = [r["loss"] for r in data["iterations"]]
        c = colors[idx % len(colors)]
        axes[0, 0].plot(iters, losses, f'{c}-o', label=f'{name} (rank={data["lora_rank"]})', markersize=4)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Loss (NLL)")
    axes[0, 0].set_title("Loss vs Iteration (Dense Model Sessions)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Session Create Times
    create_times = [sessions[n]["create_time"] for n in session_names]
    x = np.arange(len(session_names))
    colors_bar = ["blue", "red", "green", "orange", "purple"][:len(session_names)]
    axes[0, 1].bar(x, create_times, color=colors_bar, alpha=0.7)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f'{n}\n(rank={sessions[n]["lora_rank"]})' for n in session_names], fontsize=8)
    axes[0, 1].set_ylabel("Create Time (s)")
    axes[0, 1].set_title("Session Create Time (actor reuse after first)")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Iteration Time Comparison
    iter_times = {n: [r["fb_time"] + r["opt_time"] for r in sessions[n]["iterations"]] for n in session_names}
    avg_times = [np.mean(iter_times[n]) for n in session_names]
    std_times = [np.std(iter_times[n]) for n in session_names]
    axes[1, 0].bar(x, avg_times, yerr=std_times, color=colors_bar, alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f'{n}\n(rank={sessions[n]["lora_rank"]})' for n in session_names], fontsize=8)
    axes[1, 0].set_ylabel("Avg Iteration Time (s)")
    axes[1, 0].set_title("Iteration Time by Session")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Summary text
    axes[1, 1].axis('off')
    first_create = sessions[session_names[0]]["create_time"]
    reuse_times = [sessions[n]["create_time"] for n in session_names[1:]] if len(session_names) > 1 else []
    avg_reuse = np.mean(reuse_times) if reuse_times else 0
    actor_reuse = avg_reuse < 5.0 if reuse_times else True

    summary_text = f"""
    Phase 8: Dense Model Multi-Session Sharing Test

    Model: {results['model']}
    Backend: {results.get('backend', 'peft')}
    Sessions: {len(session_names)}

    Session Create Times:
    {chr(10).join([f'  {n}: {sessions[n]["create_time"]:.2f}s (rank={sessions[n]["lora_rank"]})' for n in session_names])}

    Avg Iteration Times:
    {chr(10).join([f'  {n}: {np.mean(iter_times[n]):.2f}s' for n in session_names])}

    Actor Reuse Verified: {actor_reuse}
    (First session: {first_create:.2f}s, Subsequent avg: {avg_reuse:.2f}s)

    Key Finding:
    Dense model sessions should show fast create times after first
    due to DenseTrainerPool actor reuse.
    """
    axes[1, 1].text(0.1, 0.9, summary_text, fontsize=9, family='monospace',
                    verticalalignment='top', transform=axes[1, 1].transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plots saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 8 Dense Model Sharing Test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--ranks", type=str, default="16,32,64", help="Comma-separated LoRA ranks to test")
    parser.add_argument("--num-iters", type=int, default=10, help="Iterations per session")
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase8_dense_sharing"))
    args = parser.parse_args()

    ranks = [int(r) for r in args.ranks.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("PHASE 8: DENSE MODEL MULTI-SESSION SHARING TEST")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"LoRA ranks to test: {ranks}")
    print(f"Iterations per session: {args.num_iters}")
    print(f"{'='*60}\n")

    results = {
        "model": args.model,
        "ranks": ranks,
        "num_iters": args.num_iters,
        "start_time": datetime.now().isoformat(),
        "sessions": {},
    }

    # Check initial pool status
    pool_status = get_dense_pool_status()
    print(f"Initial DenseTrainerPool status: {pool_status}\n")

    backend = None

    # Test each rank with interleaved training
    for rank in ranks:
        session_name = f"rank{rank}"
        print(f"=== Creating Session: {session_name} (LoRA rank {rank}) ===")

        session_id, model_id, create_time, sess_backend = create_session(args.model, rank, session_name)
        backend = sess_backend
        print(f"  Session: {session_id}, model_id: {model_id}")
        print(f"  Create time: {create_time:.2f}s, backend: {sess_backend}")

        # Check pool status after creation
        pool_status = get_dense_pool_status()
        print(f"  Pool status: {pool_status}")

        # Train iterations
        iterations = []
        for i in range(args.num_iters):
            # Use different token ranges for different ranks to differentiate training data
            token_base = 1000 + rank * 100
            loss, fb_time = forward_backward(model_id, token_base=token_base)
            opt_time = optim_step(model_id)
            iterations.append({
                "iteration": i + 1,
                "loss": loss,
                "fb_time": fb_time,
                "opt_time": opt_time,
            })
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  Iter {i+1}: loss={loss:.4f}, time={fb_time + opt_time:.2f}s")

        results["sessions"][session_name] = {
            "session_id": session_id,
            "model_id": model_id,
            "lora_rank": rank,
            "create_time": create_time,
            "backend": sess_backend,
            "iterations": iterations,
        }
        print()

    results["backend"] = backend

    # Verify actor reuse
    session_names = list(results["sessions"].keys())
    first_create = results["sessions"][session_names[0]]["create_time"]
    reuse_times = [results["sessions"][n]["create_time"] for n in session_names[1:]]

    actor_reuse = all(t < 5.0 for t in reuse_times) if reuse_times else True
    results["actor_reuse_verified"] = actor_reuse
    results["end_time"] = datetime.now().isoformat()

    print("="*60)
    print("ACTOR REUSE VERIFICATION")
    print("="*60)
    print(f"First session create: {first_create:.2f}s")
    for idx, name in enumerate(session_names[1:]):
        print(f"{name} create: {reuse_times[idx]:.2f}s")
    print(f"Actor reuse verified: {actor_reuse}")

    # Calculate timing consistency
    print("\n=== Iteration Time Consistency ===")
    for name in session_names:
        sess = results["sessions"][name]
        times = [r["fb_time"] + r["opt_time"] for r in sess["iterations"]]
        avg = np.mean(times)
        std = np.std(times)
        print(f"{name} (rank={sess['lora_rank']}): avg={avg:.2f}s, std={std:.3f}s")

    # Session isolation check
    print("\n=== Session Isolation ===")
    for name in session_names:
        sess = results["sessions"][name]
        initial_loss = sess["iterations"][0]["loss"]
        final_loss = sess["iterations"][-1]["loss"]
        print(f"{name}: initial={initial_loss:.4f}, final={final_loss:.4f}")

    # Loss divergence between sessions (proves isolation)
    if len(session_names) >= 2:
        losses_0 = [r["loss"] for r in results["sessions"][session_names[0]]["iterations"]]
        losses_1 = [r["loss"] for r in results["sessions"][session_names[1]]["iterations"]]
        min_len = min(len(losses_0), len(losses_1))
        avg_diff = np.mean([abs(losses_0[i] - losses_1[i]) for i in range(min_len)])
        print(f"Avg loss diff between {session_names[0]} and {session_names[1]}: {avg_diff:.4f}")
        results["avg_loss_diff"] = avg_diff

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output_dir / f"dense_sharing_test_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {output_file}")

    # Generate plots
    plot_path = args.output_dir / f"dense_sharing_test_{timestamp}.png"
    plot_results(results, plot_path)

    print(f"\n{'='*60}")
    print("PHASE 8 DENSE MODEL SHARING TEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
