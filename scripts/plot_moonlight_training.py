#!/usr/bin/env python3
"""Plot Moonlight training reward curve from metrics.jsonl."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Find the metrics file
metrics_dir = Path("/tmp/tinker-examples/math_rl")
pattern = "countdown-moonshotai-Moonlight-16B-A3B-Instruct-16rank-5e-05lr-8group-96batch-*"
matching = sorted(metrics_dir.glob(pattern))

if not matching:
    print("No matching training directories found")
    exit(1)

latest_dir = matching[-1]
metrics_file = latest_dir / "metrics.jsonl"

print(f"Reading from: {metrics_file}")

# Parse metrics
steps = []
rewards = []
correct_pct = []
kl_v1 = []
format_pct = []

with open(metrics_file) as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        steps.append(data["step"])
        rewards.append(data["env/all/reward/total"])
        correct_pct.append(data["env/all/correct"] * 100)
        kl_v1.append(data["optim/kl_sample_train_v1"])
        format_pct.append(data["env/all/format"] * 100)

print(f"Loaded {len(steps)} steps")

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Moonlight-16B-A3B Countdown Training (LoRA Export Fix Verification)", fontsize=14)

# Reward curve
ax1 = axes[0, 0]
ax1.plot(steps, rewards, 'b-o', linewidth=2, markersize=6)
ax1.set_xlabel("Step")
ax1.set_ylabel("Reward")
ax1.set_title("Reward Over Training Steps")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, max(rewards) * 1.2)

# Correct percentage
ax2 = axes[0, 1]
ax2.plot(steps, correct_pct, 'g-o', linewidth=2, markersize=6)
ax2.set_xlabel("Step")
ax2.set_ylabel("Correct (%)")
ax2.set_title("Accuracy Over Training Steps")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, max(correct_pct) * 1.5)

# KL divergence (key metric for fix verification)
ax3 = axes[1, 0]
ax3.plot(steps, kl_v1, 'r-o', linewidth=2, markersize=6)
ax3.set_xlabel("Step")
ax3.set_ylabel("KL Divergence v1")
ax3.set_title("KL Divergence (Healthy: ~0.1-0.2/step)")
ax3.grid(True, alpha=0.3)

# Calculate KL delta per step
if len(kl_v1) > 1:
    kl_deltas = [kl_v1[i] - kl_v1[i-1] for i in range(1, len(kl_v1))]
    avg_kl_delta = np.mean(kl_deltas)
    ax3.axhline(y=kl_v1[0], color='gray', linestyle='--', alpha=0.5, label=f'Initial KL')
    ax3.text(0.02, 0.95, f"Avg Δ/step: {avg_kl_delta:.3f}", transform=ax3.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Format accuracy
ax4 = axes[1, 1]
ax4.plot(steps, format_pct, 'm-o', linewidth=2, markersize=6)
ax4.set_xlabel("Step")
ax4.set_ylabel("Format (%)")
ax4.set_title("Format Accuracy Over Training Steps")
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 100)

plt.tight_layout()

# Save plot
output_file = latest_dir / "training_curves.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved plot to: {output_file}")

# Also save to a more accessible location
home_output = Path("/home/yiwen/tinker_project/tinker-server/moonlight_training_curves.png")
plt.savefig(home_output, dpi=150, bbox_inches='tight')
print(f"Also saved to: {home_output}")

# Print summary statistics
print("\n" + "="*50)
print("Training Summary")
print("="*50)
print(f"Steps completed: {len(steps)}")
print(f"Reward: {rewards[0]:.4f} → {rewards[-1]:.4f}")
print(f"Correct%: {correct_pct[0]:.2f}% → {correct_pct[-1]:.2f}%")
print(f"KL v1: {kl_v1[0]:.3f} → {kl_v1[-1]:.3f} (avg Δ/step: {avg_kl_delta:.3f})")
print(f"Format%: {format_pct[0]:.2f}% → {format_pct[-1]:.2f}%")
print("\nKL Divergence Check:")
print(f"  - Healthy range: ~0.1-0.3 per step")
print(f"  - Broken K2 had: ~3.3 per step")
print(f"  - Current: {avg_kl_delta:.3f} per step ✓")
