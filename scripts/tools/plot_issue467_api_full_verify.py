import json
from pathlib import Path

import matplotlib.pyplot as plt


RUN_DIR = Path("results/issue467_api_full_verify_20260409_retry1")
TRAIN_METRICS = RUN_DIR / "train_metrics.jsonl"
SUMMARY = RUN_DIR / "summary.json"
LOSS_PNG = RUN_DIR / "loss_curve.png"
ACC_PNG = RUN_DIR / "acc_curve.png"


def load_losses() -> tuple[list[int], list[float]]:
    steps = []
    losses = []
    with TRAIN_METRICS.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            steps.append(int(row["step"]))
            losses.append(float(row["loss_metrics"]["loss:mean"]))
    return steps, losses


def load_accs() -> tuple[list[int], list[float], list[float]]:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    checkpoint_results = payload["checkpoint_results"]
    steps = []
    train_accs = []
    test_accs = []
    for key in sorted(checkpoint_results.keys()):
        step = int(key.split("_")[1])
        steps.append(step)
        train_accs.append(float(checkpoint_results[key]["train"]["summary"]["overall_accuracy"]))
        test_accs.append(float(checkpoint_results[key]["test"]["summary"]["overall_accuracy"]))
    return steps, train_accs, test_accs


def plot_loss() -> None:
    steps, losses = load_losses()
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, linewidth=1.5, color="#1f77b4")
    for checkpoint in (0, 40, 80, 160):
        plt.axvline(checkpoint, color="#999999", linestyle="--", linewidth=0.8)
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Cross-entropy loss")
    plt.title("Issue 467 30B SFT loss curve")
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(LOSS_PNG, dpi=180)
    plt.close()


def plot_acc() -> None:
    steps, train_accs, test_accs = load_accs()
    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_accs, marker="o", linewidth=2, color="#2ca02c", label="train")
    plt.plot(steps, test_accs, marker="o", linewidth=2, color="#d62728", label="test")
    plt.xlabel("Checkpoint step")
    plt.ylabel("Accuracy")
    plt.title("Issue 467 30B SFT train/test accuracy")
    plt.xticks(steps, [str(step) for step in steps])
    plt.ylim(-0.02, 1.05)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACC_PNG, dpi=180)
    plt.close()


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss()
    plot_acc()
    print(LOSS_PNG)
    print(ACC_PNG)


if __name__ == "__main__":
    main()
