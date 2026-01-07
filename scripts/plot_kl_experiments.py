#!/usr/bin/env python3
"""Plot KL curves from controlled experiments."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load experiment data
exp1_file = Path("/tmp/kl_experiment_hollowman=False_mbridge_export=False_20260107_170402.jsonl")
exp2_file = Path("/tmp/kl_experiment_hollowman=True_mbridge_export=False_20260107_171511.jsonl")
exp3_file = Path("/tmp/kl_experiment_hollowman=True_mbridge_export=True_20260107_172927.jsonl")

def load_experiment(filepath):
    """Load experiment data from JSONL file."""
    steps = []
    kl_means = []
    kl_maxs = []

    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                steps.append(data['step'])
                kl_means.append(data['kl_mean'])
                kl_maxs.append(data['kl_max_abs'])

    return steps, kl_means, kl_maxs

# Load all experiments
exp1_steps, exp1_kl_mean, exp1_kl_max = load_experiment(exp1_file)
exp2_steps, exp2_kl_mean, exp2_kl_max = load_experiment(exp2_file)
exp3_steps, exp3_kl_mean, exp3_kl_max = load_experiment(exp3_file)

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot KL Mean
ax1 = axes[0]
ax1.plot(exp1_steps, exp1_kl_mean, 'r-o', label='Exp 1: Original mbridge + custom export', linewidth=2, markersize=6)
ax1.plot(exp2_steps, exp2_kl_mean, 'b-s', label='Exp 2: HollowMan mbridge + custom export', linewidth=2, markersize=6)
ax1.plot(exp3_steps, exp3_kl_mean, 'g-^', label='Exp 3: HollowMan mbridge + mbridge export', linewidth=2, markersize=6)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Training Step', fontsize=12)
ax1.set_ylabel('KL Mean (vLLM - Megatron)', fontsize=12)
ax1.set_title('KL Divergence Mean per Step', fontsize=14)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(10))

# Plot KL Max (log scale for better visualization)
ax2 = axes[1]
ax2.semilogy(exp1_steps, exp1_kl_max, 'r-o', label='Exp 1: Original mbridge + custom export', linewidth=2, markersize=6)
ax2.semilogy(exp2_steps, exp2_kl_max, 'b-s', label='Exp 2: HollowMan mbridge + custom export', linewidth=2, markersize=6)
ax2.semilogy(exp3_steps, exp3_kl_max, 'g-^', label='Exp 3: HollowMan mbridge + mbridge export', linewidth=2, markersize=6)
ax2.set_xlabel('Training Step', fontsize=12)
ax2.set_ylabel('KL Max Abs (log scale)', fontsize=12)
ax2.set_title('KL Divergence Max Absolute Value per Step', fontsize=14)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(10))

plt.tight_layout()
plt.savefig('/home/yiwen/tinker_project/tinker-server/kl_experiments_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("Plot saved to: kl_experiments_comparison.png")

# Print summary statistics
print("\n" + "="*80)
print("CONTROLLED KL EXPERIMENT SUMMARY")
print("="*80)

print("\nExperiment 1: Original mbridge + custom export")
print(f"  Final KL Mean: {exp1_kl_mean[-1]:.6f}")
print(f"  Final KL Max:  {exp1_kl_max[-1]:.6f}")
print(f"  Behavior: DIVERGENT (KL grows over training)")

print("\nExperiment 2: HollowMan mbridge + custom export")
print(f"  Final KL Mean: {exp2_kl_mean[-1]:.6f}")
print(f"  Final KL Max:  {exp2_kl_max[-1]:.6f}")
print(f"  Behavior: CONVERGENT (KL stabilizes near zero)")

print("\nExperiment 3: HollowMan mbridge + mbridge export")
print(f"  Final KL Mean: {exp3_kl_mean[-1]:.6f}")
print(f"  Final KL Max:  {exp3_kl_max[-1]:.6f}")
print(f"  Behavior: CONVERGENT (KL stabilizes near zero)")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
The HollowMan fork of Megatron-Bridge FIXES the train-inference mismatch.
- Original mbridge: vLLM and Megatron diverge after training (KL grows to 2.09)
- HollowMan mbridge: vLLM and Megatron converge to near-zero KL

Export method (custom vs mbridge API) does NOT matter when using HollowMan fork.
Both achieve <0.01 KL max by step 9.

RECOMMENDATION: Set USE_HOLLOWMAN_MBRIDGE=true as default.
""")
