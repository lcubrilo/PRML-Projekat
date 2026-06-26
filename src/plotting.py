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

# UFC weight classes ascending by weight: men's divisions first, then women's,
# oddities (Catch Weight) last. Use this everywhere a plot/print compares divisions.
WEIGHT_CLASS_ORDER = [
    "Flyweight", "Bantamweight", "Featherweight", "Lightweight", "Welterweight",
    "Middleweight", "Light Heavyweight", "Heavyweight",
    "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight",
    "Catch Weight",
]


def order_weight_classes(present):
    """Sort the weight classes in `present` by WEIGHT_CLASS_ORDER; unknown labels appended."""
    present = list(present)
    known = [w for w in WEIGHT_CLASS_ORDER if w in present]
    return known + [w for w in present if w not in WEIGHT_CLASS_ORDER]


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


def use_report_style():
    """Bump matplotlib font sizes so figures stay legible when embedded in the report."""
    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "figure.titlesize": 16,
    })


# Human-readable names for the difference features (raw column name shown in parens by nice_label).
HUMAN = {
    "age_dif": "Age difference", "reach_dif": "Reach difference", "height_dif": "Height difference",
    "sig_str_dif": "Avg significant strikes landed (diff)", "avg_td_dif": "Avg takedowns landed (diff)",
    "avg_sub_att_dif": "Avg submission attempts (diff)", "win_streak_dif": "Current win-streak difference",
    "lose_streak_dif": "Current lose-streak difference", "longest_win_streak_dif": "Longest win-streak (diff)",
    "win_dif": "Career wins difference", "loss_dif": "Career losses difference",
    "ko_dif": "Career KO/TKO wins (diff)", "sub_dif": "Career submission wins (diff)",
    "total_round_dif": "Total rounds fought (diff)", "total_title_bout_dif": "Career title bouts (diff)",
}


def nice_label(col, with_raw=True):
    """Human label for a column; keeps the raw name in parentheses so it stays traceable."""
    base = HUMAN.get(col) or col.replace("_dif", " difference").replace("_", " ").strip().capitalize()
    return f"{base} ({col})" if with_raw else base


def plot_decision_regions(model, X2, y, classes, xlabel="", ylabel="", ax=None,
                          resolution=300, alpha=0.30, max_points=1500, seed=0):
    """Shade a fitted 2-feature model's predicted regions, then overlay true points.

    `model` must already be fit on the same 2 columns as `X2` (n, 2). Illustrative only:
    a 2D re-fit, not the full high-dimensional model.
    """
    import numpy as np
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.4))
    classes = list(classes)
    cls_to_i = {c: i for i, c in enumerate(classes)}
    x0, x1 = X2[:, 0].min(), X2[:, 0].max()
    y0, y1 = X2[:, 1].min(), X2[:, 1].max()
    px, py = (x1 - x0) * 0.05, (y1 - y0) * 0.05
    xx, yy = np.meshgrid(np.linspace(x0 - px, x1 + px, resolution),
                         np.linspace(y0 - py, y1 + py, resolution))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Zi = np.array([cls_to_i[c] for c in Z]).reshape(xx.shape)
    cmap = plt.get_cmap("coolwarm" if len(classes) == 2 else "tab10", len(classes))
    ax.contourf(xx, yy, Zi, levels=np.arange(len(classes) + 1) - 0.5, alpha=alpha, cmap=cmap)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))[:max_points]
    for i, c in enumerate(classes):
        m = np.asarray(y)[idx] == c
        ax.scatter(X2[idx][m, 0], X2[idx][m, 1], s=8, alpha=0.55,
                   color=cmap(i), edgecolor="k", linewidth=0.2, label=str(c))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(markerscale=1.6, framealpha=0.9)
    return ax
