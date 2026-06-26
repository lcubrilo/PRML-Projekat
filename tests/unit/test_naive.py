"""Unit checks for the naive baselines (src/baselines/naive.py)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np

from src.baselines import MajorityClassifier, ConstantClassifier


def _data():
    # class 'b' is the majority (5 of 9)
    y = np.array(["a", "b", "b", "c", "b", "a", "b", "c", "b"])
    X = np.zeros((len(y), 2))
    return X, y


def test_majority_predicts_most_frequent():
    X, y = _data()
    clf = MajorityClassifier().fit(X, y)
    assert clf.majority_ == "b"
    assert np.all(clf.predict(X) == "b")
    # its accuracy equals the majority class frequency
    assert np.isclose(clf.score(X, y), np.mean(y == "b"))


def test_majority_proba_is_class_frequency():
    X, y = _data()
    clf = MajorityClassifier().fit(X, y)
    P = clf.predict_proba(X)
    assert P.shape == (len(X), len(clf.classes_))
    assert np.allclose(P.sum(axis=1), 1.0)
    # column for 'b' should be 5/9
    b = list(clf.classes_).index("b")
    assert np.isclose(P[0, b], 5 / 9)


def test_constant_classifier():
    X, y = _data()
    clf = ConstantClassifier("a").fit(X, y)
    assert np.all(clf.predict(X) == "a")
    assert np.isclose(clf.score(X, y), np.mean(y == "a"))
    P = clf.predict_proba(X)
    a = list(clf.classes_).index("a")
    assert np.allclose(P[:, a], 1.0) and np.allclose(P.sum(axis=1), 1.0)


def test_constant_rejects_unknown_label():
    X, y = _data()
    try:
        ConstantClassifier("z").fit(X, y)
        assert False, "should have raised on an unknown label"
    except ValueError:
        pass


if __name__ == '__main__':
    g = dict(globals())
    fns = [(n, f) for n, f in g.items() if n.startswith('test_') and callable(f)]
    for n, f in fns:
        f(); print('PASS', n)
    print(f'{len(fns)} unit checks passed')
