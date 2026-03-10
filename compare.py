"""Compare standard SGD vs DP-SGD on MNIST.

Trains two identical architectures -- one with vanilla backprop and one with
differentially-private backprop (Algorithm 1 from Abadi et al., 2016) -- then
prints a summary table and saves loss / accuracy curves to results.png.
"""

import matplotlib.pyplot as plt

from model import DPNeuralNetwork

# ── Hyperparameters (shared where possible) ──────────────────────────
EPOCHS = 5
LR = 0.1
BATCH_SIZE = 256

# DP-specific
NOISE_SCALE = 1.1
CLIP_BOUND = 1.0
DELTA = 1e-5


def main():
    # ── Standard SGD ─────────────────────────────────────────────────
    print("=" * 60)
    print("Training with STANDARD SGD")
    print("=" * 60)
    standard = DPNeuralNetwork(
        backprop_type="standard",
        lr=LR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    standard.load_data()
    hist_std = standard.train()

    # ── DP-SGD ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Training with DP-SGD")
    print("=" * 60)
    dp = DPNeuralNetwork(
        backprop_type="dp",
        lr=LR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        noise_scale=NOISE_SCALE,
        clip_bound=CLIP_BOUND,
        delta=DELTA,
    )
    dp.load_data()
    hist_dp = dp.train()

    eps, delta = dp.compute_privacy_cost()

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    header = f"{'':>6}  {'Train Loss':>12}  {'Test Acc':>10}  {'Time (s)':>10}"
    print(header)
    print("-" * len(header))

    final_std = hist_std[-1]
    final_dp = hist_dp[-1]
    print(
        f"{'Std':>6}  {final_std['train_loss']:>12.4f}  "
        f"{final_std['test_accuracy']:>10.4f}  "
        f"{sum(h['elapsed_sec'] for h in hist_std):>10.1f}"
    )
    print(
        f"{'DP':>6}  {final_dp['train_loss']:>12.4f}  "
        f"{final_dp['test_accuracy']:>10.4f}  "
        f"{sum(h['elapsed_sec'] for h in hist_dp):>10.1f}"
    )
    print()
    print(f"DP privacy cost:  epsilon = {eps:.2f},  delta = {delta}")
    print(f"  (noise_scale={NOISE_SCALE}, clip_bound={CLIP_BOUND}, "
          f"batch_size={BATCH_SIZE}, epochs={EPOCHS})")

    # ── Plot ─────────────────────────────────────────────────────────
    epochs_range = [h["epoch"] for h in hist_std]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs_range, [h["train_loss"] for h in hist_std], "o-", label="Standard SGD")
    ax1.plot(epochs_range, [h["train_loss"] for h in hist_dp], "s-", label="DP-SGD")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, [h["test_accuracy"] for h in hist_std], "o-", label="Standard SGD")
    ax2.plot(epochs_range, [h["test_accuracy"] for h in hist_dp], "s-", label="DP-SGD")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Accuracy")
    ax2.set_title("Test Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Standard SGD vs DP-SGD on MNIST", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("results.png", dpi=150)
    print(f"\nPlot saved to results.png")


if __name__ == "__main__":
    main()
