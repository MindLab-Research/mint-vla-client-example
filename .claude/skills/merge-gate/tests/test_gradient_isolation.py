"""Gradient Isolation Tests.

Tests that concurrent sessions have properly isolated gradients.
Uses agent-analyzed approach: collect full training data + plots for analysis.

Test methodology:
1. Create two sessions (A and B) with DIFFERENT data seeds
2. Run interleaved forward_backward calls: A.fwdbwd, B.fwdbwd, A.optim, B.optim
3. Collect FULL training curves (every iteration, all metrics)
4. Generate comparison plots for visual analysis
5. Detect anomalies (high correlation = possible gradient leakage)

Agent analyzes results both quantitatively and visually - no hardcoded thresholds.
"""

import time
import uuid
from pathlib import Path

import pytest

from .framework import (
    SessionData,
    GradientIsolationData,
    GradientAccumulationData,
    PlotGenerator,
    TestReport,
    create_test_report,
    print_report_summary,
    detect_training_anomalies,
    detect_isolation_anomalies,
    RESULTS_DIR,
)
from .conftest import (
    DENSE_MODEL,
    DENSE_SMALL_MODEL,
    MOE_MODEL,
    create_session,
    forward_backward,
    optim_step,
    train_step,
)


def make_sft_batch(tokenizer, seed: int, batch_size: int = 4) -> list[dict]:
    """Create SFT batch with deterministic seed for reproducibility.

    Different seeds produce different training data, allowing us to test
    that sessions with different data learn independently.
    """
    import random
    rng = random.Random(seed)

    # Simple arithmetic tasks with different seeds producing different numbers
    templates = [
        "Calculate: {a} + {b} = ",
        "What is {a} times {b}? Answer: ",
        "Compute {a} - {b} = ",
        "{a} divided by {b} equals ",
    ]

    data = []
    for i in range(batch_size):
        template = templates[i % len(templates)]
        a = rng.randint(1, 100)
        b = rng.randint(1, 100)
        prompt = template.format(a=a, b=b)

        # Compute answer based on operation
        if "+" in template:
            answer = str(a + b)
        elif "times" in template:
            answer = str(a * b)
        elif "-" in template:
            answer = str(a - b)
        else:
            answer = f"{a / b:.2f}"

        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)

        tokens = prompt_tokens + answer_tokens
        weights = [0.0] * len(prompt_tokens) + [1.0] * len(answer_tokens)

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        loss_mask = weights[1:]

        data.append({
            "model_input": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
                "loss_mask": {"data": loss_mask, "shape": [len(loss_mask)], "dtype": "float32"},
            },
        })

    return data


def run_gradient_isolation_test(
    base_model: str,
    tokenizer,
    num_iterations: int = 5,
    lr: float = 1e-4,
    use_train_step: bool = False,
) -> GradientIsolationData:
    """Run gradient isolation test and collect full training data.

    Args:
        base_model: Model to test.
        tokenizer: Tokenizer for the model.
        num_iterations: Number of training iterations.
        lr: Learning rate.
        use_train_step: If True, use combined train_step (for MoE with param_offload).
                       If False, use separate forward_backward + optim_step.

    Returns:
        GradientIsolationData with full training curves for both sessions.
    """
    seed_a = 42
    seed_b = 12345

    # Create two sessions
    print(f"\nCreating session A (seed={seed_a})...")
    _, model_id_a = create_session(base_model, lora_rank=32, lr=lr)
    session_a = SessionData(
        session_id=model_id_a,
        model_id=model_id_a,
        base_model=base_model,
        data_seed=seed_a,
        lora_rank=32,
        learning_rate=lr,
    )

    print(f"Creating session B (seed={seed_b})...")
    _, model_id_b = create_session(base_model, lora_rank=32, lr=lr)
    session_b = SessionData(
        session_id=model_id_b,
        model_id=model_id_b,
        base_model=base_model,
        data_seed=seed_b,
        lora_rank=32,
        learning_rate=lr,
    )

    # Prepare different data for each session
    data_a = make_sft_batch(tokenizer, seed_a, batch_size=4)
    data_b = make_sft_batch(tokenizer, seed_b, batch_size=4)

    print(f"\nRunning {num_iterations} interleaved iterations...")

    for i in range(num_iterations):
        t0 = time.time()

        if use_train_step:
            # Combined train_step for MoE (handles param_offload correctly)
            result_a = train_step(model_id_a, data_a, lr=lr, loss_fn="cross_entropy")
            result_b = train_step(model_id_b, data_b, lr=lr, loss_fn="cross_entropy")

            loss_a = result_a.get("metrics", {}).get("loss:mean", 0)
            loss_b = result_b.get("metrics", {}).get("loss:mean", 0)
            grad_norm_a = result_a.get("metrics", {}).get("grad_norm")
            grad_norm_b = result_b.get("metrics", {}).get("grad_norm")
        else:
            # Interleaved pattern: A.fwdbwd -> B.fwdbwd -> A.optim -> B.optim
            result_a = forward_backward(model_id_a, data_a, loss_fn="cross_entropy")
            result_b = forward_backward(model_id_b, data_b, loss_fn="cross_entropy")

            loss_a = result_a.get("metrics", {}).get("loss:mean", 0)
            loss_b = result_b.get("metrics", {}).get("loss:mean", 0)

            optim_a = optim_step(model_id_a, lr=lr)
            optim_b = optim_step(model_id_b, lr=lr)

            grad_norm_a = optim_a.get("metrics", {}).get("grad_norm")
            grad_norm_b = optim_b.get("metrics", {}).get("grad_norm")

        wall_time = time.time() - t0

        session_a.add_iteration(
            iteration=i + 1,
            loss=loss_a,
            grad_norm=grad_norm_a,
            wall_time_seconds=wall_time / 2,  # Approximate split
        )
        session_b.add_iteration(
            iteration=i + 1,
            loss=loss_b,
            grad_norm=grad_norm_b,
            wall_time_seconds=wall_time / 2,
        )

        print(f"  Iter {i+1}: A.loss={loss_a:.4f}, B.loss={loss_b:.4f}, time={wall_time:.2f}s")

    return GradientIsolationData(
        session_a=session_a,
        session_b=session_b,
        interleave_pattern="A.fwdbwd -> B.fwdbwd -> A.optim -> B.optim" if not use_train_step else "A.train_step -> B.train_step",
    )


def run_gradient_accumulation_test(
    base_model: str,
    tokenizer,
    accumulation_count: int = 3,
    lr: float = 1e-4,
    use_train_step: bool = False,
) -> GradientAccumulationData:
    """Run gradient accumulation test and collect training data.

    Compares:
    - Single forward_backward + optim_step
    - Multiple forward_backward + single optim_step (gradient accumulation)

    Args:
        base_model: Model to test.
        tokenizer: Tokenizer for the model.
        accumulation_count: Number of forward_backward calls to accumulate.
        lr: Learning rate.
        use_train_step: If True, can't test accumulation (train_step is atomic).

    Returns:
        GradientAccumulationData with training curves for comparison.
    """
    if use_train_step:
        # Can't test accumulation with train_step - skip for MoE
        print("Skipping accumulation test for MoE (train_step is atomic)")
        return None

    seed = 42
    data = make_sft_batch(tokenizer, seed, batch_size=4)

    # Session 1: Single forward_backward + optim_step
    print("\nSession 1: Single forward_backward per optim_step...")
    _, model_id_1 = create_session(base_model, lora_rank=32, lr=lr)
    single_session = SessionData(
        session_id=model_id_1,
        model_id=model_id_1,
        base_model=base_model,
        data_seed=seed,
        lora_rank=32,
        learning_rate=lr,
    )

    # Initial loss
    result = forward_backward(model_id_1, data, loss_fn="cross_entropy")
    initial_loss = result.get("metrics", {}).get("loss:mean", 0)
    single_session.add_iteration(iteration=0, loss=initial_loss)

    optim_step(model_id_1, lr=lr)

    # Post-step loss
    result = forward_backward(model_id_1, data, loss_fn="cross_entropy")
    post_loss = result.get("metrics", {}).get("loss:mean", 0)
    single_session.add_iteration(iteration=1, loss=post_loss)

    print(f"  Single: {initial_loss:.4f} -> {post_loss:.4f}")

    # Session 2: Multiple forward_backward + single optim_step
    print(f"\nSession 2: {accumulation_count}x forward_backward before optim_step...")
    _, model_id_2 = create_session(base_model, lora_rank=32, lr=lr)
    multi_session = SessionData(
        session_id=model_id_2,
        model_id=model_id_2,
        base_model=base_model,
        data_seed=seed,
        lora_rank=32,
        learning_rate=lr,
    )

    # Initial loss
    result = forward_backward(model_id_2, data, loss_fn="cross_entropy")
    initial_loss_2 = result.get("metrics", {}).get("loss:mean", 0)
    multi_session.add_iteration(iteration=0, loss=initial_loss_2)

    # Accumulate gradients
    for j in range(1, accumulation_count):
        forward_backward(model_id_2, data, loss_fn="cross_entropy")

    # Single optim step with accumulated gradients
    optim_step(model_id_2, lr=lr)

    # Post-step loss
    result = forward_backward(model_id_2, data, loss_fn="cross_entropy")
    post_loss_2 = result.get("metrics", {}).get("loss:mean", 0)
    multi_session.add_iteration(iteration=1, loss=post_loss_2)

    print(f"  Accumulated ({accumulation_count}x): {initial_loss_2:.4f} -> {post_loss_2:.4f}")

    return GradientAccumulationData(
        single_fwdbwd=single_session,
        multi_fwdbwd=multi_session,
        accumulation_count=accumulation_count,
    )


class TestGradientIsolationDense:
    """Gradient isolation tests for Dense models."""

    def test_isolation_dense(self, tokenizer):
        """Test gradient isolation between concurrent Dense sessions.

        Agent-analyzed test:
        - Collects FULL training curves for both sessions
        - Generates comparison plots for visual analysis
        - Detects anomalies (high correlation may indicate gradient leakage)
        - NO hardcoded pass/fail thresholds
        """
        start_time = time.time()

        # Run test and collect data
        data = run_gradient_isolation_test(
            base_model=DENSE_MODEL,
            tokenizer=tokenizer,
            num_iterations=5,
            lr=1e-4,
            use_train_step=False,
        )

        # Generate visualizations
        plotter = PlotGenerator()
        plot_path = plotter.gradient_isolation_comparison(data)
        plots = [plot_path] if plot_path else []

        # Also generate individual training curves
        curve_a = plotter.training_curve(
            data.session_a,
            title=f"Session A (seed={data.session_a.data_seed})",
            filename=f"dense_isolation_session_a_{time.strftime('%Y%m%d_%H%M%S')}.png",
        )
        curve_b = plotter.training_curve(
            data.session_b,
            title=f"Session B (seed={data.session_b.data_seed})",
            filename=f"dense_isolation_session_b_{time.strftime('%Y%m%d_%H%M%S')}.png",
        )
        if curve_a:
            plots.append(curve_a)
        if curve_b:
            plots.append(curve_b)

        # Create test report
        report = create_test_report(
            test_name="dense_gradient_isolation",
            test_type="isolation",
            data=data,
            start_time=start_time,
            plots=plots,
            metadata={
                "model": DENSE_MODEL,
                "num_iterations": 5,
                "learning_rate": 1e-4,
            },
        )

        # Save report
        report_path = report.save()
        print_report_summary(report)

        # Output for agent analysis
        print(f"\n{'='*70}")
        print("AGENT ANALYSIS REQUIRED")
        print(f"{'='*70}")
        print(f"Report: {report_path}")
        if plots:
            print(f"Plots:")
            for p in plots:
                print(f"  - {p}")
        print(f"\nCorrelation: {data.compute_correlation():.3f}")
        print(f"Session A reduction: {data.session_a.loss_reduction_pct:.1f}%")
        print(f"Session B reduction: {data.session_b.loss_reduction_pct:.1f}%")

        if report.anomalies:
            print(f"\nANOMALIES ({len(report.anomalies)}):")
            for a in report.anomalies:
                print(f"  - {a}")

        # No hardcoded assertions - agent analyzes results
        # However, flag extreme issues for attention
        if any("Invalid loss" in a for a in report.anomalies):
            pytest.fail("NaN/Inf loss detected - training unstable")

    def test_accumulation_dense(self, tokenizer):
        """Test gradient accumulation for Dense models.

        Verifies that multiple forward_backward calls accumulate gradients
        correctly before a single optim_step.
        """
        start_time = time.time()

        # Run test and collect data
        data = run_gradient_accumulation_test(
            base_model=DENSE_MODEL,
            tokenizer=tokenizer,
            accumulation_count=3,
            lr=1e-4,
            use_train_step=False,
        )

        if data is None:
            pytest.skip("Accumulation test not applicable")

        # Generate visualization
        plotter = PlotGenerator()
        plot_path = plotter.gradient_accumulation_comparison(data)
        plots = [plot_path] if plot_path else []

        # Create test report
        report = create_test_report(
            test_name="dense_gradient_accumulation",
            test_type="accumulation",
            data=data,
            start_time=start_time,
            plots=plots,
            metadata={
                "model": DENSE_MODEL,
                "accumulation_count": 3,
                "learning_rate": 1e-4,
            },
        )

        report_path = report.save()
        print_report_summary(report)

        # Output for agent analysis
        print(f"\n{'='*70}")
        print("AGENT ANALYSIS REQUIRED")
        print(f"{'='*70}")
        print(f"Report: {report_path}")
        print(f"Single drop: {data.single_drop:.4f}")
        print(f"Accumulated drop ({data.accumulation_count}x): {data.accumulated_drop:.4f}")

        if data.single_drop and data.single_drop > 0:
            ratio = data.accumulated_drop / data.single_drop
            print(f"Effect ratio: {ratio:.2f}x")


class TestGradientIsolationMoE:
    """Gradient isolation tests for MoE models."""

    def test_isolation_moe(self, moe_tokenizer):
        """Test gradient isolation between concurrent MoE sessions.

        Uses train_step (combined forward_backward + optim_step) because
        MoE with param_offload requires single train_mode context.
        """
        start_time = time.time()

        # Run test with train_step (MoE requires this for param_offload)
        data = run_gradient_isolation_test(
            base_model=MOE_MODEL,
            tokenizer=moe_tokenizer,
            num_iterations=5,
            lr=1e-4,
            use_train_step=True,
        )

        # Generate visualizations
        plotter = PlotGenerator()
        plot_path = plotter.gradient_isolation_comparison(data)
        plots = [plot_path] if plot_path else []

        # Create test report
        report = create_test_report(
            test_name="moe_gradient_isolation",
            test_type="isolation",
            data=data,
            start_time=start_time,
            plots=plots,
            metadata={
                "model": MOE_MODEL,
                "num_iterations": 5,
                "learning_rate": 1e-4,
                "note": "Using train_step for MoE param_offload",
            },
        )

        report_path = report.save()
        print_report_summary(report)

        # Output for agent analysis
        print(f"\n{'='*70}")
        print("AGENT ANALYSIS REQUIRED")
        print(f"{'='*70}")
        print(f"Report: {report_path}")
        print(f"Correlation: {data.compute_correlation():.3f}")
        red_a = data.session_a.loss_reduction_pct
        red_b = data.session_b.loss_reduction_pct
        print(f"Session A reduction: {red_a:.1f}%" if red_a is not None else "Session A reduction: n/a")
        print(f"Session B reduction: {red_b:.1f}%" if red_b is not None else "Session B reduction: n/a")

        if report.anomalies:
            print(f"\nANOMALIES ({len(report.anomalies)}):")
            for a in report.anomalies:
                print(f"  - {a}")


class TestGradientIsolationSmall:
    """Quick gradient isolation tests with small model (for CI)."""

    def test_isolation_small(self, small_tokenizer):
        """Quick isolation test with Qwen3-0.6B.

        Faster than full Dense model, suitable for CI.
        """
        start_time = time.time()

        data = run_gradient_isolation_test(
            base_model=DENSE_SMALL_MODEL,
            tokenizer=small_tokenizer,
            num_iterations=3,  # Fewer iterations for speed
            lr=1e-4,
            use_train_step=False,
        )

        plotter = PlotGenerator()
        plot_path = plotter.gradient_isolation_comparison(data)
        plots = [plot_path] if plot_path else []

        report = create_test_report(
            test_name="small_gradient_isolation",
            test_type="isolation",
            data=data,
            start_time=start_time,
            plots=plots,
            metadata={
                "model": DENSE_SMALL_MODEL,
                "num_iterations": 3,
                "learning_rate": 1e-4,
                "purpose": "CI quick test",
            },
        )

        report_path = report.save()
        print_report_summary(report)

        print(f"\nReport: {report_path}")
        print(f"Correlation: {data.compute_correlation():.3f}")
