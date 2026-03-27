"""
Visualization helpers for the classification / localization pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import CLASS_NAMES, FIGURES_DIR


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> None:
    """Display a heatmap confusion matrix and optionally save to disk."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Sample prediction grid
# ---------------------------------------------------------------------------

def plot_sample_predictions(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str] = CLASS_NAMES,
    n: int = 16,
    save_path: str | Path | None = None,
) -> None:
    """Show a grid of sample images with true/predicted labels.

    Correct predictions are shown in green, wrong ones in red.
    """
    n = min(n, len(images))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()

    for i in range(n):
        ax = axes[i]
        img = images[i]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        correct = y_true[i] == y_pred[i]
        color = "green" if correct else "red"
        ax.set_title(
            f"T:{class_names[y_true[i]]} P:{class_names[y_pred[i]]}",
            color=color,
            fontsize=9,
        )
        ax.axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Model comparison chart
# ---------------------------------------------------------------------------

def plot_model_comparison(
    results: dict[str, dict[str, float]],
    save_path: str | Path | None = None,
) -> None:
    """Grouped bar chart comparing metrics across models."""
    import pandas as pd

    df = pd.DataFrame(results).T

    # Scale figure width to keep configuration labels readable.
    n_models = len(df.index)
    max_label_len = max((len(str(label)) for label in df.index), default=0)
    fig_width = max(10, min(22, n_models * 0.9))
    rotate_labels = n_models > 8 or max_label_len > 14
    label_rotation = 35 if rotate_labels else 0

    ax = df.plot.bar(figsize=(fig_width, 5), rot=label_rotation)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend(loc="lower right")
    if rotate_labels:
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), ha="right")
    fig = ax.get_figure()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Training history (Option C)
# ---------------------------------------------------------------------------

def plot_training_history(
    history: dict[str, list[float]],
    save_path: str | Path | None = None,
) -> None:
    """Plot train/val loss and accuracy curves.

    Parameters
    ----------
    history : dict with keys like ``train_loss``, ``val_loss``,
              ``train_acc``, ``val_acc`` — each a list of per-epoch values.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train")
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train")
    if "val_acc" in history:
        ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()