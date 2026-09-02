import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_history(log_path: str, output_path: str):
    with open(log_path) as file:
        history = json.load(file)
    if not history:
        raise ValueError("The training log contains no epochs.")

    epochs = [row["epoch"] for row in history]
    plots = [
        ("loss", "Total loss"),
        ("bce", "Weighted BCE loss"),
        ("mse", "Attention adjacency MSE"),
        ("dice", "Dice score"),
        ("iou", "IoU score"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Dental Caries ViT Training Curves", fontsize=15, fontweight="bold")

    for axis, (key, title) in zip(axes.flat[:5], plots):
        axis.plot(epochs, [row["train"][key] for row in history], label="Train", linewidth=2)
        axis.plot(epochs, [row["val"][key] for row in history], label="Validation", linewidth=2)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        axis.legend()

    lr_axis = axes.flat[5]
    lr_axis.set_title("Learning rate")
    lr_axis.set_xlabel("Epoch")
    learning_rates = [row.get("lr") for row in history]
    if all(rate is not None for rate in learning_rates):
        lr_axis.plot(epochs, learning_rates, color="tab:green", linewidth=2)
        lr_axis.set_yscale("log")
        lr_axis.grid(alpha=0.3)
    else:
        lr_axis.text(0.5, 0.5, "Learning-rate history\nnot available in this run", ha="center", va="center",
                     transform=lr_axis.transAxes)
        lr_axis.set_xticks([])
        lr_axis.set_yticks([])

    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training metrics from training_log.json")
    parser.add_argument("--log", required=True, help="Path to training_log.json")
    parser.add_argument("--output", default="training_curves.png")
    args = parser.parse_args()
    plot_history(args.log, args.output)
