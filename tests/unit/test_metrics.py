"""Validate the from-scratch metrics (src/metrics.py) against sklearn.metrics on
a real classifier's predictions and probabilities. sklearn is the reference only.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as skm

from src.metrics import (accuracy, confusion_matrix, precision_recall_f1,
                         log_loss, brier_score, roc_auc_ovr, summarize_over_seeds)


def _fit():
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    return yte, clf.predict(Xte), proba, clf.classes_


def test_accuracy_matches():
    yte, yp, _, _ = _fit()
    assert np.isclose(accuracy(yte, yp), skm.accuracy_score(yte, yp))


def test_confusion_matrix_matches():
    yte, yp, _, classes = _fit()
    M, _ = confusion_matrix(yte, yp, classes=classes)
    assert np.array_equal(M, skm.confusion_matrix(yte, yp, labels=classes))


def test_macro_f1_matches():
    yte, yp, _, _ = _fit()
    ours = precision_recall_f1(yte, yp)["macro_f1"]
    assert np.isclose(ours, skm.f1_score(yte, yp, average="macro"))


def test_log_loss_matches():
    yte, _, proba, classes = _fit()
    assert np.isclose(log_loss(yte, proba, classes=classes), skm.log_loss(yte, proba))


def test_brier_matches():
    # sklearn has no multiclass Brier; compare to the explicit one-hot MSE definition
    yte, _, proba, classes = _fit()
    onehot = np.eye(len(classes))[np.searchsorted(classes, yte)]
    expected = np.mean(np.sum((proba - onehot) ** 2, axis=1))
    assert np.isclose(brier_score(yte, proba, classes=classes), expected)


def test_roc_auc_ovr_matches():
    yte, _, proba, classes = _fit()
    ours = roc_auc_ovr(yte, proba, classes=classes)
    ref = skm.roc_auc_score(yte, proba, multi_class="ovr", average="macro")
    assert np.isclose(ours, ref, atol=1e-6)


def test_summarize_over_seeds():
    m, s = summarize_over_seeds([0.6, 0.7, 0.8])
    assert np.isclose(m, 0.7) and np.isclose(s, np.std([0.6, 0.7, 0.8]))


if __name__ == '__main__':
    g = dict(globals())
    fns = [(n, f) for n, f in g.items() if n.startswith('test_') and callable(f)]
    for n, f in fns:
        f(); print('PASS', n)
    print(f'{len(fns)} unit checks passed')
