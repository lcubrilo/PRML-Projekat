"""Linear discriminant classifiers — ported from `03_least_squares`.

Perceptron (single-sample), Ho–Kashyap, and the least-squares/MSE classifier.
All use the course's augment-and-normalize trick (append 1, negate class 2),
so they are inherently **2-class**; use one-vs-rest for multiclass. Faithful to
the course bodies, wrapped sklearn-style for binary labels.

Primary implementations; sklearn used only in tests.
"""
import numpy as np


def augment_and_normalize(class1, class2):
    """Append a 1 (bias) to each sample; negate class-2 rows (course trick)."""
    Y1 = np.column_stack([class1, np.ones(len(class1))])
    Y2 = -np.column_stack([class2, np.ones(len(class2))])
    return np.vstack([Y1, Y2])


def perceptron_single_sample(Y, eta=1.0, max_epochs=50, a_init=None):
    d = Y.shape[1]
    a = np.zeros(d) if a_init is None else a_init.astype(float).copy()
    history = [a.copy()]
    for _ in range(max_epochs):
        updated = False
        for y in Y:
            if a @ y <= 0:
                a = a + eta * y
                updated = True
                history.append(a.copy())
        if not updated:
            break
    return a, history


def ho_kashyap(Y, eta=0.5, max_iters=1000, tol=1e-3):
    n, d = Y.shape
    a = np.zeros(d)
    b = np.ones(n)
    for _ in range(max_iters):
        e = Y @ a - b
        b = b + eta * (e + np.abs(e))
        a = np.linalg.pinv(Y) @ b
        if np.all(np.abs(e) < tol):
            break
    return a, b


def mse_classification_solution(Y, b=None):
    """Least-squares (pseudoinverse) solution to Y a ≈ b."""
    if b is None:
        b = np.ones(Y.shape[0])
    return np.linalg.pinv(Y) @ b


class _BinaryLinear:
    """Shared sklearn-style wrapper. Builds Y from the two classes, fits `a`,
    predicts by sign of a·[x,1]. classes_[0] is the negated (class-2) label."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2, "binary only (wrap in one-vs-rest for multiclass)"
        c1 = X[y == self.classes_[1]]   # positive class
        c2 = X[y == self.classes_[0]]   # negated class
        Y = augment_and_normalize(c1, c2)
        self.a_ = self._solve(Y)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        Xa = np.column_stack([X, np.ones(len(X))])
        return Xa @ self.a_

    def predict(self, X):
        return np.where(self.decision_function(X) > 0, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


class PerceptronClassifier(_BinaryLinear):
    def __init__(self, eta=1.0, max_epochs=50):
        self.eta, self.max_epochs = eta, max_epochs
    def _solve(self, Y):
        return perceptron_single_sample(Y, self.eta, self.max_epochs)[0]


class HoKashyapClassifier(_BinaryLinear):
    def __init__(self, eta=0.5, max_iters=1000):
        self.eta, self.max_iters = eta, max_iters
    def _solve(self, Y):
        return ho_kashyap(Y, self.eta, self.max_iters)[0]


class LeastSquaresClassifier(_BinaryLinear):
    def _solve(self, Y):
        return mse_classification_solution(Y)
