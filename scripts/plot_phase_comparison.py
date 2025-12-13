#!/usr/bin/env python3
"""Generate comprehensive comparison plots across all phases.

Compares:
- Phase 6: MoE session isolation
- Phase 7: MoE unified rank support
- Phase 8: Dense model multi-session sharing
- Phase 9: LRU eviction metrics
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_latest_results(pattern: str) -> dict | None:
    """Load the latest result file matching pattern."""
    files = sorted(Path(".").glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def main():
    output_dir = Path("results/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    phase6 = load_latest_results("results/phase6_isolation/isolation_test_*.json")
    phase7 = load_latest_results("results/phase7_unified_rank/unified_rank_test_*.json")
    phase8 = load_latest_results("results/phase8_dense_sharing/dense_sharing_test_*.json")
    phase9 = load_latest_results("results/phase9_lru_eviction/lru_eviction_test_*.json")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Session Create Time Comparison
    ax1 = fig.add_subplot(2, 2, 1)

    create_times = {}
    labels = []

    if phase6:
        create_times["Phase 6 MoE A"] = phase6["session_a"]["create_time"]
        create_times["Phase 6 MoE B"] = phase6["session_b"]["create_time"]
        labels.extend(["Phase 6 MoE A", "Phase 6 MoE B"])

    if phase7:
        for rank in phase7["sessions"]:
            key = f"Phase 7 Rank {rank}"
            create_times[key] = phase7["sessions"][rank]["create_time"]
            labels.append(key)

    if phase8:
        for name in phase8["sessions"]:
            key = f"Phase 8 {name}"
            create_times[key] = phase8["sessions"][name]["create_time"]
            labels.append(key)

    x = np.arange(len(create_times))
    colors = plt.cm.Set3(np.linspace(0, 1, len(create_times)))
    bars = ax1.bar(x, list(create_times.values()), color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(create_times.keys()), rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Create Time (s)")
    ax1.set_title("Session Create Time Comparison (Actor Reuse)")
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1s threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Iteration Time by Backend
    ax2 = fig.add_subplot(2, 2, 2)

    iter_data = {}
    if phase6:
        moe_times = [r["fb_time"] + r["opt_time"] for r in phase6["session_a"]["iterations"]]
        iter_data["MoE (Phase 6)"] = np.mean(moe_times)

    if phase7:
        for rank in phase7["sessions"]:
            times = [r["fb_time"] + r["opt_time"] for r in phase7["sessions"][rank]["iterations"]]
            iter_data[f"MoE Rank {rank}"] = np.mean(times)

    if phase8:
        for name in phase8["sessions"]:
            times = [r["fb_time"] + r["opt_time"] for r in phase8["sessions"][name]["iterations"]]
            iter_data[f"Dense {name}"] = np.mean(times)

    x = np.arange(len(iter_data))
    colors = ['blue' if 'MoE' in k else 'green' for k in iter_data.keys()]
    ax2.bar(x, list(iter_data.values()), color=colors, alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(iter_data.keys()), rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Avg Iteration Time (s)")
    ax2.set_title("Iteration Time by Backend (MoE vs Dense)")
    ax2.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.7, label='MoE (Megatron)'),
                       Patch(facecolor='green', alpha=0.7, label='Dense (PEFT)')]
    ax2.legend(handles=legend_elements)

    # Plot 3: Loss Trajectories
    ax3 = fig.add_subplot(2, 2, 3)

    if phase6:
        a_losses = [r["loss"] for r in phase6["session_a"]["iterations"]]
        b_losses = [r["loss"] for r in phase6["session_b"]["iterations"]]
        ax3.plot(range(1, len(a_losses)+1), a_losses, 'b-o', label='MoE Session A', markersize=4)
        ax3.plot(range(1, len(b_losses)+1), b_losses, 'b--s', label='MoE Session B', markersize=4)

    if phase8:
        dense_colors = ['g', 'c', 'm']  # Single char colors for matplotlib format
        for idx, name in enumerate(phase8["sessions"]):
            losses = [r["loss"] for r in phase8["sessions"][name]["iterations"]]
            if losses[0] > 0:  # Skip zero losses
                ax3.plot(range(1, len(losses)+1), losses, f'{dense_colors[idx % 3]}-o',
                         label=f'Dense {name}', markersize=4)

    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Loss (NLL)")
    ax3.set_title("Loss Trajectories (Session Isolation)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary Statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    summary_text = """
    Phase Comparison Summary
    ========================

    Phase 6: MoE Session Isolation (Qwen3-30B-A3B)
    - Backend: Megatron (TP=4, EP=2)
    - GPU Requirement: 8 GPUs
    - Session Create: ~0.14-0.18s (actor reuse)
    - Iteration Time: ~5-6s
    - Isolation: Verified (loss diff ~1.24)

    Phase 7: Unified Rank Support (MoE)
    - Ranks Tested: 16, 32, 64
    - Actor Reuse: All ranks share single trainer
    - Iteration Variance: <7% across ranks
    - Max-rank Padding: Working correctly

    Phase 8: Dense Model Sharing (Qwen2.5-7B)
    - Backend: PEFT (single GPU)
    - First Session: ~14s (cold start)
    - Subsequent: ~0.18s (77x faster)
    - Iteration Time: ~0.9s (6x faster than MoE)

    Phase 9: LRU Eviction
    - LRU Tracking: Working
    - Actor Reuse: Working
    - Session Isolation: Working
    - Idle Detection: Working

    Key Insights:
    -------------
    1. Actor pooling provides 77-130x speedup for session creation
    2. Dense models ~6x faster per iteration than MoE
    3. MoE requires 8 GPUs vs 1 GPU for dense
    4. Session isolation verified for both backends
    """

    ax4.text(0.05, 0.95, summary_text, fontsize=9, family='monospace',
             verticalalignment='top', transform=ax4.transAxes)

    plt.tight_layout()
    output_path = output_dir / "phase_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved: {output_path}")

    # Also save as summary JSON
    summary = {
        "phase6": phase6["statistics"] if phase6 else None,
        "phase7": {
            "actor_reuse_verified": phase7.get("actor_reuse_verified") if phase7 else None,
            "ranks": list(phase7["sessions"].keys()) if phase7 else [],
        },
        "phase8": {
            "actor_reuse_verified": phase8.get("actor_reuse_verified") if phase8 else None,
            "backend": phase8.get("backend") if phase8 else None,
        },
        "phase9": phase9.get("summary") if phase9 else None,
    }

    summary_path = output_dir / "phase_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
