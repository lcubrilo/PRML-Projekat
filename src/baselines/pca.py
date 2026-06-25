"""PCA — ported from `09_pca/9_PCA_solution.ipynb`.

Course recipe: center → covariance (Xc.T @ Xc / n) → eigendecomposition →
sort eigenpairs descending → project / reconstruct → explained-variance ratio.
The course used np.linalg.eig (its own comment hints '# eigh'); since the
covariance is symmetric we use np.linalg.eigh (real spectrum, more stable) and
reverse to descending order — same result, cleaner numerics.

Primary implementation; sklearn used only in tests.
"""
import numpy as np


def center_data(X):
    mean_vec = np.mean(X, axis=0)
    return X - mean_vec, mean_vec


def covariance_matrix(X_centered):
    n = X_centered.shape[0]
    return (X_centered.T @ X_centered) / n


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        Xc, self.mean_ = center_data(X)
        S = covariance_matrix(Xc)
        eigvals, eigvecs = np.linalg.eigh(S)            # ascending, real
        idx = np.argsort(eigvals)[::-1]                 # → descending
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        self.explained_variance_ = eigvals
        self.explained_variance_ratio_ = eigvals / np.sum(eigvals)
        k = self.n_components if self.n_components is not None else X.shape[1]
        self.components_ = eigvecs[:, :k]               # columns = PCs
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return np.asarray(Z, dtype=float) @ self.components_.T + self.mean_
