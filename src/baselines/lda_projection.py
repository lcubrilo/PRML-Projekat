"""Fisher LDA as *dimensionality reduction* - ported from `10_lda`.

Distinct from the LDA *classifier* in `gaussian.py` (shared-covariance Gaussian):
this is the supervised projection that maximizes between/within-class scatter
(eigenvectors of SW^-1 SB), the natural partner to PCA for a "does DR help?" story.

Primary implementation; sklearn used only in tests.
"""
import numpy as np


def class_means(X, y):
    return {c: X[y == c].mean(axis=0) for c in np.unique(y)}


def within_class_scatter(X, y, means):
    d = X.shape[1]
    SW = np.zeros((d, d))
    for c in np.unique(y):
        for x in X[y == c]:
            diff = (x - means[c]).reshape(-1, 1)
            SW += diff @ diff.T
    return SW


def between_class_scatter(X, y, means, mu):
    d = X.shape[1]
    SB = np.zeros((d, d))
    for c in np.unique(y):
        nc = np.sum(y == c)
        diff = (means[c] - mu).reshape(-1, 1)
        SB += nc * (diff @ diff.T)
    return SB


class LDAProjection:
    """Project onto the top-(n_components) Fisher directions. Max rank = C-1."""

    def __init__(self, n_components=1):
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        means = class_means(X, y)
        mu = X.mean(axis=0)
        SW = within_class_scatter(X, y, means)
        SB = between_class_scatter(X, y, means, mu)
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(SW) @ SB)
        eigvals, eigvecs = np.real(eigvals), np.real(eigvecs)
        idx = np.argsort(eigvals)[::-1]
        W = eigvecs[:, idx[: self.n_components]]
        self.components_ = W / np.linalg.norm(W, axis=0)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) @ self.components_

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
