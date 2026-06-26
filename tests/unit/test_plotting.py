"""Smoke tests for the plotting helpers: they should build axes without error.
Uses the non-interactive Agg backend and does not write any files.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.plotting import plot_confusion_matrix, plot_sweep, plot_staged_accuracy


def test_confusion_matrix_builds():
    M = np.array([[5, 1], [2, 4]])
    ax = plot_confusion_matrix(M, classes=["x", "y"])
    assert ax is not None
    ax2 = plot_confusion_matrix(M, classes=["x", "y"], normalize=True)
    assert ax2 is not None


def test_sweep_builds():
    ax = plot_sweep([1, 3, 5], [0.6, 0.65, 0.62], xlabel="k", label="kNN")
    assert ax.get_xlabel() == "k"


def test_staged_accuracy_builds():
    ax = plot_staged_accuracy([0.4, 0.55, 0.6, 0.62])
    assert ax.get_xlabel() == "boosting rounds"


if __name__ == '__main__':
    g = dict(globals())
    fns = [(n, f) for n, f in g.items() if n.startswith('test_') and callable(f)]
    for n, f in fns:
        f(); print('PASS', n)
    print(f'{len(fns)} unit checks passed')
