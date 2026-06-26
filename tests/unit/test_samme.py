"""Validation for the from-scratch SAMME extension (src/extension/samme.py).

This file is the executable spec: it will FAIL until Luka implements the
stub. Code samme.py until `python tests/unit/test_samme.py` prints PASS.

We validate against sklearn's AdaBoostClassifier(algorithm='SAMME') with
depth-1 trees - the exact algorithm we reimplement. Note we do NOT require
identical predictions: our stump's threshold/tie-breaking can differ from
sklearn's tree, so we assert (a) high prediction agreement and (b) accuracy
within a small tolerance. sklearn is used ONLY here, never for reported numbers.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from src.extension import SAMMEClassifier, DecisionStump


# --- helpers ---------------------------------------------------------------
def _iris(binary=False):
    X, y = load_iris(return_X_y=True)
    if binary:                       # classes 0 and 1 only -> K=2, SAMME == AdaBoost
        mask = y < 2
        X, y = X[mask], y[mask]
    ys = np.array(["a", "b", "c"])[y]    # arbitrary string labels, like test_knn
    return train_test_split(X, ys, test_size=0.3, random_state=0, stratify=y)


def _ref(n_estimators):
    """sklearn discrete-SAMME reference. Built defensively across sklearn
    versions (the `algorithm`/`base_estimator` kwargs have churned)."""
    base = DecisionTreeClassifier(max_depth=1)
    for kwargs in (
        dict(estimator=base, algorithm="SAMME"),   # modern sklearn
        dict(estimator=base),                       # algorithm removed (SAMME is default)
        dict(base_estimator=base, algorithm="SAMME"),  # older sklearn
    ):
        try:
            return AdaBoostClassifier(n_estimators=n_estimators, random_state=0, **kwargs)
        except TypeError:
            continue
    raise RuntimeError("could not construct an AdaBoost SAMME reference")


def _acc(pred, y):
    return float(np.mean(pred == y))


# --- tests -----------------------------------------------------------------
def test_stump_matches_sklearn():
    """The base learner alone must match a weighted depth-1 tree."""
    Xtr, Xte, ytr, yte = _iris()
    _, yenc = np.unique(ytr, return_inverse=True)
    rng = np.random.default_rng(0)
    w = rng.random(len(ytr)); w /= w.sum()

    ours = DecisionStump().fit(Xtr, yenc, w)
    ref = DecisionTreeClassifier(max_depth=1, random_state=0).fit(Xtr, yenc, sample_weight=w)
    agree = np.mean(ours.predict(Xte) == ref.predict(Xte))
    assert agree > 0.9, f"stump disagrees with sklearn tree ({agree:.2f})"


def test_multiclass_matches_sklearn():
    """3-class iris: exercises the log(K-1) term. Compare to sklearn SAMME."""
    Xtr, Xte, ytr, yte = _iris()
    for T in (10, 50):
        ours = SAMMEClassifier(n_estimators=T).fit(Xtr, ytr)
        ref = _ref(T).fit(Xtr, ytr)
        agree = np.mean(ours.predict(Xte) == ref.predict(Xte))
        assert agree > 0.85, f"T={T}: low agreement with sklearn ({agree:.2f})"
        assert abs(_acc(ours.predict(Xte), yte) - _acc(ref.predict(Xte), yte)) < 0.07


def test_binary_reduces_to_adaboost():
    """K=2 -> log(K-1)=0 -> SAMME is exactly AdaBoost. Should fit cleanly."""
    Xtr, Xte, ytr, yte = _iris(binary=True)
    ours = SAMMEClassifier(n_estimators=50).fit(Xtr, ytr)
    assert _acc(ours.predict(Xte), yte) > 0.9    # iris setosa/versicolor is separable


def test_predict_proba_is_valid():
    """E1 needs probabilities for log-loss / ROC-AUC."""
    Xtr, Xte, ytr, yte = _iris()
    clf = SAMMEClassifier(n_estimators=30).fit(Xtr, ytr)
    P = clf.predict_proba(Xte)
    assert P.shape == (len(Xte), len(clf.classes_))
    assert np.all(P >= 0)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
    # argmax of proba must agree with predict
    assert np.all(clf.classes_[np.argmax(P, axis=1)] == clf.predict(Xte))


def test_score_matches_predict():
    Xtr, Xte, ytr, yte = _iris()
    clf = SAMMEClassifier(n_estimators=20).fit(Xtr, ytr)
    assert np.isclose(clf.score(Xte, yte), _acc(clf.predict(Xte), yte))


def test_staged_score_ends_at_full_ensemble():
    """staged_score's last entry must equal the full-ensemble score (E2 curve)."""
    Xtr, Xte, ytr, yte = _iris()
    clf = SAMMEClassifier(n_estimators=30).fit(Xtr, ytr)
    staged = clf.staged_score(Xte, yte)
    assert len(staged) == len(clf.stumps_)
    assert np.isclose(staged[-1], clf.score(Xte, yte))


if __name__ == '__main__':
    g = dict(globals())
    fns = [(n, f) for n, f in g.items() if n.startswith('test_') and callable(f)]
    for n, f in fns:
        f(); print('PASS', n)
    print(f'{len(fns)} unit checks passed')
