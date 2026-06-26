"""From-scratch evaluation metrics for the classification results.

Implemented directly in NumPy so the project stays self-contained; sklearn.metrics
appears only in tests, to check these against a reference. Label-classification
metrics take (y_true, y_pred); probabilistic metrics take a probability matrix
whose columns are aligned to `classes` (pass `model.classes_` so the alignment is
explicit).
"""
import numpy as np


def _index_map(classes):
    return {c: i for i, c in enumerate(classes)}


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, classes=None):
    """Counts with rows = true class, columns = predicted class, in `classes` order.

    Returns (matrix, classes).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    idx = _index_map(classes)
    yt = np.array([idx[c] for c in y_true])
    yp = np.array([idx[c] for c in y_pred])
    M = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(M, (yt, yp), 1)
    return M, np.asarray(classes)


def precision_recall_f1(y_true, y_pred, classes=None) -> dict:
    """Per-class precision/recall/F1 plus their macro averages."""
    M, classes = confusion_matrix(y_true, y_pred, classes)
    tp = np.diag(M).astype(float)
    fp = M.sum(axis=0) - tp
    fn = M.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {
        "classes": classes,
        "precision": precision, "recall": recall, "f1": f1,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }


def log_loss(y_true, proba, classes=None, eps=1e-15) -> float:
    """Multiclass cross-entropy. `proba` is (n_samples, n_classes) aligned to `classes`."""
    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true)
    if classes is None:
        classes = np.unique(y_true)
    idx = _index_map(classes)
    yt = np.array([idx[c] for c in y_true])
    p = np.clip(proba[np.arange(len(yt)), yt], eps, 1.0)
    return float(-np.mean(np.log(p)))


def brier_score(y_true, proba, classes=None) -> float:
    """Multiclass Brier score: mean squared error between proba and one-hot truth."""
    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true)
    if classes is None:
        classes = np.unique(y_true)
    idx = _index_map(classes)
    yt = np.array([idx[c] for c in y_true])
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(yt)), yt] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _average_ranks(x):
    """1-based ranks of x with ties broken by averaging (as in the AUC formula)."""
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(len(x), dtype=float)
    i, n = 0, len(x)
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0  # average rank, 1-based
        i = j + 1
    return ranks


def roc_auc_ovr(y_true, proba, classes=None) -> float:
    """Macro one-vs-rest ROC-AUC via the Mann-Whitney rank statistic (ties averaged)."""
    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true)
    if classes is None:
        classes = np.unique(y_true)
    aucs = []
    for i, c in enumerate(classes):
        pos = y_true == c
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = _average_ranks(proba[:, i])
        auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        aucs.append(auc)
    return float(np.mean(aucs))


def summarize_over_seeds(values):
    """Mean and population std over repeated runs. Returns (mean, std)."""
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std())


def collapse_winner(labels):
    """Map joint 'Red-KO'/'Blue-DEC'... labels down to the winner ('Red'/'Blue')."""
    return np.array([str(s).split("-")[0] for s in labels])


def collapse_method(labels):
    """Map joint labels down to the method ('KO'/'SUB'/'DEC')."""
    return np.array([str(s).split("-")[1] for s in labels])
