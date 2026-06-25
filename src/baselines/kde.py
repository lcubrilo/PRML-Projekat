"""Kernel Density Estimation + a Parzen-window classifier — from `07_kde`.

NOTE: the course `07_kde` notebook is the *un-solved* version; its `kde_estimate`
has the `n` vs `len(samples)` bug the TA flagged on Moodle, `silverman_bandwidth`
is a `pass` stub, and `epanechnikov_kernel` has a `==`/`=` typo. This port FIXES
all three. The Parzen classifier (class-conditional KDE → Bayes rule) is the
density-based baseline.

Primary implementation; sklearn used only in tests.
"""
import numpy as np


def uniform_kernel(u):
    return 0.5 * (np.abs(u) <= 1)


def gaussian_kernel(u):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u ** 2)


def epanechnikov_kernel(u):
    vals = 0.75 * (1 - u ** 2)
    vals[np.abs(u) > 1] = 0.0          # FIX: assignment, was `== 0`
    return vals


def kde_estimate(x_eval, samples, h, kernel_fn=gaussian_kernel):
    """1D KDE. FIX: divide by len(samples), not a stray global `n`."""
    samples = np.asarray(samples, dtype=float)
    x_eval = np.atleast_1d(np.asarray(x_eval, dtype=float))
    n = len(samples)
    est = np.zeros_like(x_eval)
    for i, x in enumerate(x_eval):
        u = (x - samples) / h
        est[i] = np.sum(kernel_fn(u)) / (n * h)
    return est


def silverman_bandwidth(samples):
    """Silverman's rule of thumb (implemented; course left it as TODO)."""
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    sigma = np.std(samples, ddof=1)
    iqr = np.subtract(*np.percentile(samples, [75, 25]))
    spread = min(sigma, iqr / 1.349) if iqr > 0 else sigma
    return 0.9 * spread * n ** (-1 / 5)


class ParzenClassifier:
    """Multivariate Parzen-window classifier: per-class product-Gaussian KDE
    (diagonal bandwidth h per feature) combined with class priors via Bayes."""

    def __init__(self, h=1.0):
        self.h = h

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        self._Xc = {c: X[y == c] for c in self.classes_}
        self._prior = {c: len(self._Xc[c]) / len(X) for c in self.classes_}
        return self

    def _log_density(self, X, Xc):
        # product of 1D Gaussian KDEs across features (diagonal kernel)
        n, d = Xc.shape
        h = self.h
        out = np.zeros(len(X))
        for i, x in enumerate(X):
            u = (x - Xc) / h                       # (n, d)
            k = gaussian_kernel(u).prod(axis=1)    # product over features
            out[i] = np.log(k.sum() / (n * h ** d) + 1e-300)
        return out

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        scores = np.column_stack([
            self._log_density(X, self._Xc[c]) + np.log(self._prior[c]) for c in self.classes_
        ])
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
