"""Shared figure-saving helper so EVERY plot lands in one place: report/figures/.

Usage in any notebook/script:
    from src.plotting import save_fig
    ...plot...
    save_fig("class_balance")          # -> report/figures/class_balance.png

The report references these as `figures/NAME.png` (paths relative to report/report.md).
Figures ARE committed (they go into the report), so report/figures/ is NOT gitignored.
"""
from pathlib import Path
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parents[1] / "report" / "figures"


def save_fig(name: str, dpi: int = 150, tight: bool = True):
    """Save the current matplotlib figure as report/figures/<name>.png."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / f"{name}.png", dpi=dpi,
                bbox_inches="tight" if tight else None)
    return FIG_DIR / f"{name}.png"


def plot_confusion_matrix(matrix, classes, normalize: bool = False, ax=None):
    """Heatmap of a confusion matrix (rows = true, cols = predicted).

    Pass the matrix and class labels from `src.metrics.confusion_matrix`. Returns
    the Axes; call `save_fig(...)` afterwards to write it out.
    """
    import numpy as np
    M = np.asarray(matrix, dtype=float)
    if normalize:
        M = M / M.sum(axis=1, keepdims=True).clip(min=1e-12)
    if ax is None:
        _, ax = plt.subplots(figsize=(1.2 * len(classes) + 1, 1.2 * len(classes) + 1))
    im = ax.imshow(M, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    thresh = M.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = f"{M[i, j]:.2f}" if normalize else f"{int(round(M[i, j]))}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if M[i, j] > thresh else "black", fontsize=8)
    return ax


def plot_sweep(xs, ys, xlabel: str, ylabel: str = "accuracy", label=None, ax=None):
    """Line plot for a hyperparameter sweep (e.g. accuracy vs k, or vs n_estimators)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if label:
        ax.legend()
    return ax


def plot_staged_accuracy(scores, ax=None, label=None):
    """Convergence curve: accuracy after each boosting round (SAMME.staged_score)."""
    return plot_sweep(range(1, len(scores) + 1), scores,
                      xlabel="boosting rounds", ylabel="accuracy", label=label, ax=ax)
