#!/usr/bin/env python3
"""Plot MoE LR comparison results showing optimizer state fix (Issue 6c)."""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_results(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)

def plot_lr_comparison(data: dict, output_dir: Path, title_suffix: str = ""):
    """Plot loss vs iteration and loss vs wall time for all learning rates."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'0.0001': 'blue', '1e-05': 'green', '1e-06': 'red'}
    labels = {'0.0001': 'LR=1e-4', '1e-05': 'LR=1e-5', '1e-06': 'LR=1e-6'}

    # Plot 1: Loss vs Iteration
    ax1 = axes[0, 0]
    for lr_key, lr_data in data.items():
        if 'error' in lr_data:
            continue
        results = lr_data.get('results', [])
        # Filter out zero-loss iterations (session issues)
        iters = [r['iter'] for r in results if r['loss'] > 0]
        losses = [r['loss'] for r in results if r['loss'] > 0]
        if iters:
            ax1.plot(iters, losses, 'o-', color=colors.get(lr_key, 'black'),
                    label=labels.get(lr_key, lr_key), markersize=3, linewidth=1)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Iteration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Loss vs Wall Time
    ax2 = axes[0, 1]
    for lr_key, lr_data in data.items():
        if 'error' in lr_data:
            continue
        results = lr_data.get('results', [])
        times = [r['elapsed'] for r in results if r['loss'] > 0]
        losses = [r['loss'] for r in results if r['loss'] > 0]
        if times:
            ax2.plot(times, losses, 'o-', color=colors.get(lr_key, 'black'),
                    label=labels.get(lr_key, lr_key), markersize=3, linewidth=1)
    ax2.set_xlabel('Wall Time (s)')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss vs Wall Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Plot 3: Gradient Norm vs Iteration
    ax3 = axes[1, 0]
    for lr_key, lr_data in data.items():
        if 'error' in lr_data:
            continue
        results = lr_data.get('results', [])
        iters = [r['iter'] for r in results if r['grad_norm'] > 0]
        grad_norms = [r['grad_norm'] for r in results if r['grad_norm'] > 0]
        if iters:
            ax3.plot(iters, grad_norms, 'o-', color=colors.get(lr_key, 'black'),
                    label=labels.get(lr_key, lr_key), markersize=3, linewidth=1)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Gradient Norm')
    ax3.set_title('Gradient Norm vs Iteration')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_data = []
    headers = ['LR', 'Init Loss', 'Final Loss', 'Min Loss', 'Change %', 'Avg Grad']

    for lr_key in ['0.0001', '1e-05', '1e-06']:
        if lr_key not in data or 'error' in data[lr_key]:
            continue
        lr_data = data[lr_key]
        results = [r for r in lr_data.get('results', []) if r['loss'] > 0]
        if not results:
            continue

        init_loss = results[0]['loss']
        final_loss = results[-1]['loss']
        min_loss = min(r['loss'] for r in results)
        change_pct = (init_loss - final_loss) / init_loss * 100
        avg_grad = np.mean([r['grad_norm'] for r in results if r['grad_norm'] > 0])

        table_data.append([
            labels.get(lr_key, lr_key),
            f'{init_loss:.4f}',
            f'{final_loss:.4f}',
            f'{min_loss:.4f}',
            f'{change_pct:.1f}%',
            f'{avg_grad:.2f}'
        ])

    if table_data:
        table = ax4.table(cellText=table_data, colLabels=headers,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Summary Statistics', pad=20)

    plt.suptitle(f'MoE LR Comparison - Optimizer State Fix Verification{title_suffix}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'lr_comparison_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()

def plot_before_after_comparison(before_data: dict, after_data: dict, output_dir: Path):
    """Plot before/after comparison showing the fix effect."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'0.0001': 'blue', '1e-05': 'green', '1e-06': 'red'}
    labels = {'0.0001': 'LR=1e-4', '1e-05': 'LR=1e-5', '1e-06': 'LR=1e-6'}

    # Before fix
    ax1 = axes[0]
    for lr_key, lr_data in before_data.items():
        if 'error' in lr_data:
            continue
        results = lr_data.get('results', [])
        iters = [r['iter'] for r in results[:20]]  # First 20 iterations
        losses = [r['loss'] for r in results[:20]]
        if iters:
            ax1.plot(iters, losses, 'o-', color=colors.get(lr_key, 'black'),
                    label=labels.get(lr_key, lr_key), markersize=4, linewidth=1.5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('BEFORE Fix: Instant Convergence Bug\n(Optimizer state carries over)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_ylim([0.001, 10])

    # After fix
    ax2 = axes[1]
    for lr_key, lr_data in after_data.items():
        if 'error' in lr_data:
            continue
        results = lr_data.get('results', [])
        results = [r for r in results if r['loss'] > 0][:20]  # First 20 valid iterations
        iters = [r['iter'] for r in results]
        losses = [r['loss'] for r in results]
        if iters:
            ax2.plot(iters, losses, 'o-', color=colors.get(lr_key, 'black'),
                    label=labels.get(lr_key, lr_key), markersize=4, linewidth=1.5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('AFTER Fix: Gradual Descent\n(Fresh optimizer state each session)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.set_ylim([0.001, 10])

    plt.suptitle('Issue 6c Fix: Adam Optimizer State Reset Between Sessions',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'before_after_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.close()

def main():
    results_dir = Path('/home/yiwen/tinker_project/tinker-server/results/moe_lr_comparison')

    # Find latest result file (after fix)
    after_fix_file = results_dir / 'lr_comparison_20251215_101616.json'

    # Before fix file (showing the bug)
    before_fix_file = results_dir / 'lr_comparison_20251215_091648.json'

    if not after_fix_file.exists():
        print(f"Error: {after_fix_file} not found")
        sys.exit(1)

    # Load and plot after-fix results
    after_data = load_results(after_fix_file)
    plot_lr_comparison(after_data, results_dir, " (After Fix)")

    # If before-fix file exists, create comparison
    if before_fix_file.exists():
        before_data = load_results(before_fix_file)
        plot_before_after_comparison(before_data, after_data, results_dir)

    print("Plots generated in:", results_dir)

if __name__ == '__main__':
    main()
