"""Agent-Analyzed Testing Framework.

This module provides infrastructure for tests that:
1. Collect FULL training data (every iteration, all metrics)
2. Generate visualizations for agent inspection
3. Output structured reports for analysis

Tests do NOT use hardcoded pass/fail thresholds. Instead, they produce
rich data + plots that an agent analyzes to determine correctness.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np


# Results directory
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Data Collection Dataclasses
# =============================================================================

@dataclass
class IterationRecord:
    """Single training iteration metrics."""
    iteration: int
    loss: float
    grad_norm: float | None = None
    wall_time_seconds: float = 0.0
    # RL-specific
    reward: float | None = None
    accuracy: float | None = None
    ratio_mean: float | None = None
    clip_frac: float | None = None


@dataclass
class SessionData:
    """Training data for a single session."""
    session_id: str
    model_id: str
    base_model: str
    data_seed: int | None = None
    lora_rank: int = 32
    learning_rate: float = 1e-4
    iterations: list[IterationRecord] = field(default_factory=list)

    def add_iteration(self, **kwargs):
        """Add an iteration record."""
        self.iterations.append(IterationRecord(**kwargs))

    @property
    def losses(self) -> list[float]:
        return [it.loss for it in self.iterations]

    @property
    def grad_norms(self) -> list[float]:
        return [it.grad_norm for it in self.iterations if it.grad_norm is not None]

    @property
    def initial_loss(self) -> float | None:
        return self.iterations[0].loss if self.iterations else None

    @property
    def final_loss(self) -> float | None:
        return self.iterations[-1].loss if self.iterations else None

    @property
    def loss_reduction_pct(self) -> float | None:
        if self.initial_loss and self.final_loss and self.initial_loss > 0:
            return (self.initial_loss - self.final_loss) / self.initial_loss * 100
        return None


@dataclass
class GradientIsolationData:
    """Data for gradient isolation test."""
    session_a: SessionData
    session_b: SessionData
    interleave_pattern: str = "A.fwdbwd -> B.fwdbwd -> A.optim -> B.optim"

    def compute_correlation(self) -> float:
        """Compute Pearson correlation between A and B losses."""
        if len(self.session_a.losses) != len(self.session_b.losses):
            return float('nan')
        a = np.array(self.session_a.losses)
        b = np.array(self.session_b.losses)
        if len(a) < 2:
            return float('nan')
        return float(np.corrcoef(a, b)[0, 1])


@dataclass
class GradientAccumulationData:
    """Data for gradient accumulation test."""
    single_fwdbwd: SessionData  # 1x forward_backward + optim_step
    multi_fwdbwd: SessionData   # Nx forward_backward + 1x optim_step
    accumulation_count: int = 3

    @property
    def single_drop(self) -> float | None:
        if len(self.single_fwdbwd.losses) >= 2:
            return self.single_fwdbwd.losses[0] - self.single_fwdbwd.losses[1]
        return None

    @property
    def accumulated_drop(self) -> float | None:
        if len(self.multi_fwdbwd.losses) >= 2:
            return self.multi_fwdbwd.losses[0] - self.multi_fwdbwd.losses[-1]
        return None


@dataclass
class ActorSnapshot:
    """Snapshot of actor registry state."""
    timestamp: float
    phase: str
    actors: list[str]
    gpu_usage: int
    action: str = ""


@dataclass
class LRUEvictionData:
    """Data for LRU eviction test."""
    snapshots: list[ActorSnapshot] = field(default_factory=list)
    eviction_events: list[dict] = field(default_factory=list)

    def add_snapshot(self, phase: str, actors: list[str], gpu_usage: int, action: str = ""):
        self.snapshots.append(ActorSnapshot(
            timestamp=time.time(),
            phase=phase,
            actors=actors,
            gpu_usage=gpu_usage,
            action=action,
        ))


@dataclass
class CheckpointFidelityData:
    """Data for checkpoint save/load fidelity test."""
    model: str
    loss_before_save: float
    loss_after_load: float
    checkpoint_path: str
    delta: float = 0.0

    def __post_init__(self):
        self.delta = abs(self.loss_before_save - self.loss_after_load)


@dataclass
class TestReport:
    """Complete test report with all data and metadata."""
    test_id: str
    test_name: str
    test_type: Literal[
        "training",
        "isolation",
        "accumulation",
        "eviction",
        "checkpoint",
        "concurrent",
        "api",
        "long_context",
        "stress",
    ]
    timestamp: str
    duration_seconds: float = 0.0
    data: Any = None  # Test-specific data (SessionData, GradientIsolationData, etc.)
    plots: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def save(self, results_dir: Path = RESULTS_DIR) -> Path:
        """Save report to JSON file."""
        # Convert dataclasses to dicts for JSON serialization
        def to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: to_dict(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, list):
                return [to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        report_dict = to_dict(self)
        filepath = results_dir / f"{self.test_name}_{self.timestamp}.json"
        with open(filepath, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
        return filepath


# =============================================================================
# Visualization
# =============================================================================

class PlotGenerator:
    """Generate plots for agent analysis."""

    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.results_dir = results_dir
        self._check_matplotlib()

    def _check_matplotlib(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            self.plt = plt
            self.available = True
        except ImportError:
            self.available = False

    def training_curve(
        self,
        session: SessionData,
        title: str = None,
        filename: str = None,
    ) -> Path | None:
        """Generate training curve plot."""
        if not self.available:
            return None

        losses = session.losses
        iterations = list(range(1, len(losses) + 1))

        fig, axes = self.plt.subplots(2, 1, figsize=(12, 8))

        # Loss curve
        ax = axes[0]
        ax.plot(iterations, losses, 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(title or f"Training Curve: {session.base_model}")
        ax.grid(True, alpha=0.3)

        # Annotate
        if losses:
            reduction = session.loss_reduction_pct or 0
            ax.annotate(f"Initial: {losses[0]:.4f}", xy=(1, losses[0]),
                       xytext=(1.5, losses[0]), fontsize=9)
            ax.annotate(f"Final: {losses[-1]:.4f}", xy=(len(losses), losses[-1]),
                       xytext=(len(losses)-0.5, losses[-1]), fontsize=9)
            ax.text(0.02, 0.98, f"Reduction: {reduction:.1f}%",
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Gradient norm
        ax = axes[1]
        grad_norms = session.grad_norms
        if grad_norms:
            ax.plot(iterations[:len(grad_norms)], grad_norms, 'g-o', linewidth=2, markersize=6)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Gradient Norm")
            ax.set_title("Gradient Norm Over Training")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No gradient norm data", transform=ax.transAxes,
                   ha='center', va='center', fontsize=12, color='gray')

        self.plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"training_{session.session_id[:8]}_{timestamp}.png"
        filepath = self.results_dir / filename
        self.plt.savefig(filepath, dpi=150, bbox_inches='tight')
        self.plt.close()

        return filepath

    def rl_training_curves(
        self,
        session: SessionData,
        title: str | None = None,
        filename: str | None = None,
    ) -> Path | None:
        """Generate RL training plots (loss/reward/accuracy/ratio)."""
        if not self.available:
            return None

        iterations = list(range(1, len(session.iterations) + 1))
        losses = session.losses
        rewards = [it.reward for it in session.iterations]
        accuracies = [it.accuracy for it in session.iterations]
        ratios = [it.ratio_mean for it in session.iterations]

        fig, axes = self.plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.plot(iterations, losses, "b-o", linewidth=2, markersize=6)
        ax.set_title("Loss")
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        if any(r is not None for r in rewards):
            ax.plot(iterations, rewards, "g-o", linewidth=2, markersize=6)
        else:
            ax.text(0.5, 0.5, "No reward data", transform=ax.transAxes, ha="center", va="center", color="gray")
        ax.set_title("Reward")
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        if any(a is not None for a in accuracies):
            ax.plot(iterations, accuracies, "orange", linewidth=2, marker="o", markersize=6)
            ax.set_ylim(0.0, 1.0)
        else:
            ax.text(0.5, 0.5, "No accuracy data", transform=ax.transAxes, ha="center", va="center", color="gray")
        ax.set_title("Accuracy")
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if any(r is not None for r in ratios):
            ax.plot(iterations, ratios, "purple", linewidth=2, marker="o", markersize=6)
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        else:
            ax.text(0.5, 0.5, "No ratio data", transform=ax.transAxes, ha="center", va="center", color="gray")
        ax.set_title("Policy Ratio Mean")
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)

        fig.suptitle(title or f"RL Curves: {session.base_model}", fontsize=12)
        self.plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"rl_{session.session_id[:8]}_{timestamp}.png"
        filepath = self.results_dir / filename
        self.plt.savefig(filepath, dpi=150, bbox_inches="tight")
        self.plt.close()

        return filepath

    def gradient_isolation_comparison(
        self,
        data: GradientIsolationData,
        filename: str = None,
    ) -> Path | None:
        """Generate gradient isolation comparison plot (2x2 panel)."""
        if not self.available:
            return None

        fig, axes = self.plt.subplots(2, 2, figsize=(14, 10))

        losses_a = data.session_a.losses
        losses_b = data.session_b.losses
        iterations = list(range(1, max(len(losses_a), len(losses_b)) + 1))

        # Top-left: Session A loss curve
        ax = axes[0, 0]
        ax.plot(iterations[:len(losses_a)], losses_a, 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(f"Session A (seed={data.session_a.data_seed})")
        ax.grid(True, alpha=0.3)
        if losses_a:
            red_a = data.session_a.loss_reduction_pct or 0
            ax.text(0.02, 0.98, f"Reduction: {red_a:.1f}%",
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # Top-right: Session B loss curve
        ax = axes[0, 1]
        ax.plot(iterations[:len(losses_b)], losses_b, 'r-o', linewidth=2, markersize=6)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(f"Session B (seed={data.session_b.data_seed})")
        ax.grid(True, alpha=0.3)
        if losses_b:
            red_b = data.session_b.loss_reduction_pct or 0
            ax.text(0.02, 0.98, f"Reduction: {red_b:.1f}%",
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

        # Bottom-left: Both curves overlaid
        ax = axes[1, 0]
        ax.plot(iterations[:len(losses_a)], losses_a, 'b-o', linewidth=2, markersize=6, label='Session A')
        ax.plot(iterations[:len(losses_b)], losses_b, 'r-o', linewidth=2, markersize=6, label='Session B')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Both Sessions Overlaid")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Bottom-right: Correlation scatter
        ax = axes[1, 1]
        min_len = min(len(losses_a), len(losses_b))
        if min_len > 1:
            ax.scatter(losses_a[:min_len], losses_b[:min_len], c=range(min_len), cmap='viridis', s=100)
            corr = data.compute_correlation()
            ax.set_xlabel("Session A Loss")
            ax.set_ylabel("Session B Loss")
            ax.set_title(f"Correlation: {corr:.3f}")
            ax.grid(True, alpha=0.3)
            # Add colorbar for iteration
            sm = self.plt.cm.ScalarMappable(cmap='viridis', norm=self.plt.Normalize(1, min_len))
            sm.set_array([])
            cbar = self.plt.colorbar(sm, ax=ax)
            cbar.set_label('Iteration')
        else:
            ax.text(0.5, 0.5, "Insufficient data for correlation",
                   transform=ax.transAxes, ha='center', va='center')

        self.plt.suptitle(f"Gradient Isolation: {data.interleave_pattern}", fontsize=12)
        self.plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"gradient_isolation_{timestamp}.png"
        filepath = self.results_dir / filename
        self.plt.savefig(filepath, dpi=150, bbox_inches='tight')
        self.plt.close()

        return filepath

    def gradient_accumulation_comparison(
        self,
        data: GradientAccumulationData,
        filename: str = None,
    ) -> Path | None:
        """Generate gradient accumulation comparison plot."""
        if not self.available:
            return None

        fig, axes = self.plt.subplots(1, 2, figsize=(14, 5))

        # Left: Loss curves
        ax = axes[0]
        single_losses = data.single_fwdbwd.losses
        multi_losses = data.multi_fwdbwd.losses

        ax.plot(range(1, len(single_losses)+1), single_losses, 'b-o',
                linewidth=2, markersize=8, label='1x fwdbwd')
        ax.plot(range(1, len(multi_losses)+1), multi_losses, 'g-s',
                linewidth=2, markersize=8, label=f'{data.accumulation_count}x fwdbwd')
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: Bar chart of drops
        ax = axes[1]
        drops = [data.single_drop or 0, data.accumulated_drop or 0]
        labels = ['1x fwdbwd', f'{data.accumulation_count}x fwdbwd']
        colors = ['blue', 'green']
        bars = ax.bar(labels, drops, color=colors, alpha=0.7)
        ax.set_ylabel("Loss Drop")
        ax.set_title("Effect of Gradient Accumulation")

        # Annotate bars
        for bar, drop in zip(bars, drops):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{drop:.4f}', ha='center', va='bottom', fontsize=10)

        self.plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"gradient_accumulation_{timestamp}.png"
        filepath = self.results_dir / filename
        self.plt.savefig(filepath, dpi=150, bbox_inches='tight')
        self.plt.close()

        return filepath

    def lru_eviction_timeline(
        self,
        data: LRUEvictionData,
        filename: str = None,
    ) -> Path | None:
        """Generate LRU eviction Gantt-style timeline."""
        if not self.available or not data.snapshots:
            return None

        fig, ax = self.plt.subplots(figsize=(14, 6))

        # Get all unique actors
        all_actors = set()
        for snapshot in data.snapshots:
            all_actors.update(snapshot.actors)
        actors = sorted(all_actors)

        # Track actor lifetimes
        actor_presence = {actor: [] for actor in actors}
        phases = [s.phase for s in data.snapshots]

        for i, snapshot in enumerate(data.snapshots):
            for actor in actors:
                actor_presence[actor].append(1 if actor in snapshot.actors else 0)

        # Plot as horizontal bars
        colors = self.plt.cm.tab10(range(len(actors)))
        for idx, (actor, presence) in enumerate(actor_presence.items()):
            for i, p in enumerate(presence):
                if p:
                    ax.barh(idx, 1, left=i, height=0.6, color=colors[idx], alpha=0.7)

        ax.set_yticks(range(len(actors)))
        ax.set_yticklabels(actors, fontsize=8)
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel("Phase")
        ax.set_ylabel("Actor")
        ax.set_title("Actor Lifetime Timeline (LRU Eviction)")

        # Add GPU usage on secondary axis
        ax2 = ax.twinx()
        gpu_usage = [s.gpu_usage for s in data.snapshots]
        ax2.plot(range(len(gpu_usage)), gpu_usage, 'k--', linewidth=2, marker='s', label='GPU Usage')
        ax2.set_ylabel("GPU Usage", color='black')
        ax2.legend(loc='upper right')

        self.plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"lru_eviction_{timestamp}.png"
        filepath = self.results_dir / filename
        self.plt.savefig(filepath, dpi=150, bbox_inches='tight')
        self.plt.close()

        return filepath


# =============================================================================
# Anomaly Detection
# =============================================================================

def detect_training_anomalies(session: SessionData) -> list[str]:
    """Detect anomalies in training curve."""
    anomalies = []
    losses = session.losses

    if not losses:
        return ["No loss data collected"]

    # Check for NaN/Inf
    for i, loss in enumerate(losses):
        if np.isnan(loss) or np.isinf(loss):
            anomalies.append(f"Invalid loss at iteration {i+1}: {loss}")

    # Check for loss spikes (>10% increase)
    for i in range(1, len(losses)):
        prev = losses[i - 1]
        cur = losses[i]
        if cur > prev * 1.1:
            if prev > 0:
                pct = (cur / prev - 1) * 100
                anomalies.append(f"Loss spike at iter {i+1}: {prev:.4f} -> {cur:.4f} (+{pct:.1f}%)")
            else:
                anomalies.append(f"Loss spike at iter {i+1}: {prev:.4f} -> {cur:.4f} (+inf%)")

    # Check for plateau
    if len(losses) >= 3:
        recent = losses[-3:]
        spread = max(recent) - min(recent)
        avg = sum(recent) / len(recent)
        if avg > 0 and spread / avg < 0.01:
            anomalies.append(f"Loss plateau: last 3 values within {spread:.4f}")

    # Check for insufficient decrease (observation, not failure)
    if len(losses) >= 2 and losses[0] > 0:
        decrease_pct = (losses[0] - losses[-1]) / losses[0] * 100
        if decrease_pct < 10:
            anomalies.append(f"Minimal learning: only {decrease_pct:.1f}% decrease")

    return anomalies


def detect_isolation_anomalies(data: GradientIsolationData) -> list[str]:
    """Detect anomalies in gradient isolation test."""
    anomalies = []

    # Check each session
    anomalies.extend([f"[Session A] {a}" for a in detect_training_anomalies(data.session_a)])
    anomalies.extend([f"[Session B] {a}" for a in detect_training_anomalies(data.session_b)])

    # Check correlation (high correlation may indicate shared state)
    corr = data.compute_correlation()
    if not np.isnan(corr) and abs(corr) > 0.9:
        anomalies.append(f"High loss correlation: {corr:.3f} (possible shared state)")

    # Check if final losses are suspiciously identical
    if (data.session_a.final_loss is not None and
        data.session_b.final_loss is not None and
        abs(data.session_a.final_loss - data.session_b.final_loss) < 0.001):
        anomalies.append("Final losses nearly identical (possible shared state)")

    return anomalies


# =============================================================================
# Test Runner Helpers
# =============================================================================

def create_test_report(
    test_name: str,
    test_type: str,
    data: Any,
    start_time: float,
    plots: list[Path] = None,
    metadata: dict = None,
) -> TestReport:
    """Create a test report with collected data."""
    import uuid

    # Detect anomalies based on data type
    anomalies = []
    if isinstance(data, SessionData):
        anomalies = detect_training_anomalies(data)
    elif isinstance(data, GradientIsolationData):
        anomalies = detect_isolation_anomalies(data)

    report = TestReport(
        test_id=str(uuid.uuid4()),
        test_name=test_name,
        test_type=test_type,
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        duration_seconds=time.time() - start_time,
        data=data,
        plots=[str(p) for p in (plots or []) if p],
        anomalies=anomalies,
        metadata=metadata or {},
    )

    return report


def print_report_summary(report: TestReport):
    """Print human-readable report summary."""
    print(f"\n{'='*70}")
    print(f"TEST REPORT: {report.test_name}")
    print(f"{'='*70}")
    print(f"  Test ID: {report.test_id[:8]}...")
    print(f"  Type: {report.test_type}")
    print(f"  Duration: {report.duration_seconds:.1f}s")

    # Print data-specific summary
    data = report.data
    if isinstance(data, SessionData):
        print(f"  Model: {data.base_model}")
        print(f"  Iterations: {len(data.iterations)}")
        if data.initial_loss and data.final_loss:
            print(f"  Loss: {data.initial_loss:.4f} -> {data.final_loss:.4f}")
            print(f"  Reduction: {data.loss_reduction_pct:.1f}%")

    elif isinstance(data, GradientIsolationData):
        print(f"  Pattern: {data.interleave_pattern}")
        red_a = data.session_a.loss_reduction_pct
        red_b = data.session_b.loss_reduction_pct
        if red_a is None:
            print("  Session A: n/a reduction")
        else:
            print(f"  Session A: {red_a:.1f}% reduction")
        if red_b is None:
            print("  Session B: n/a reduction")
        else:
            print(f"  Session B: {red_b:.1f}% reduction")
        print(f"  Correlation: {data.compute_correlation():.3f}")

    elif isinstance(data, GradientAccumulationData):
        print(f"  Accumulation: {data.accumulation_count}x")
        if data.single_drop and data.accumulated_drop:
            print(f"  Single drop: {data.single_drop:.4f}")
            print(f"  Accumulated drop: {data.accumulated_drop:.4f}")
            ratio = data.accumulated_drop / data.single_drop if data.single_drop else 0
            print(f"  Effect ratio: {ratio:.2f}x")

    # Print anomalies
    print(f"\n  Anomalies: {len(report.anomalies)}")
    if report.anomalies:
        for a in report.anomalies:
            print(f"    - {a}")

    # Print plots
    if report.plots:
        print(f"\n  Plots generated:")
        for p in report.plots:
            print(f"    - {p}")

    print(f"{'='*70}\n")
