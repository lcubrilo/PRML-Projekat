import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.baselines import PerceptronClassifier, LeastSquaresClassifier

def _sep():
    X, y = load_iris(return_X_y=True); m = y < 2  # setosa vs versicolor (separable)
    return train_test_split(X[m], y[m], test_size=0.3, random_state=0, stratify=y[m])

def test_perceptron_separates():
    Xtr, Xte, ytr, yte = _sep()
    assert PerceptronClassifier().fit(Xtr, ytr).score(Xte, yte) == 1.0

def test_least_squares_classifier():
    Xtr, Xte, ytr, yte = _sep()
    assert LeastSquaresClassifier().fit(Xtr, ytr).score(Xte, yte) > 0.95


if __name__ == '__main__':
    g = dict(globals())
    fns = [(n, f) for n, f in g.items() if n.startswith('test_') and callable(f)]
    for n, f in fns:
        f(); print('PASS', n)
    print(f'{len(fns)} unit checks passed')
