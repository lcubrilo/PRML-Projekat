"""Linear regression - ported from `02_linear_regression` and `03_least_squares`.

Covers the two course routes that the project may compare:
  - closed-form normal equation  (course `direct_solution` / `least_squares_solution`)
  - batch gradient descent       (course `gradient_descent`, vectorized)
  - ridge (L2) closed form       (course `ridge_solution`) - bias not penalized.

A bias column is added internally. Primary implementation; sklearn used only in tests.
"""
import numpy as np


def _add_bias(X):
    return np.column_stack([np.ones(len(X)), X])


def polynomial_features(x, degree):
    """Design matrix [1, x, x^2, ..., x^degree] for scalar input (course 02)."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([x ** d for d in range(degree + 1)])


def normal_equation(Xb, t, lam=0.0):
    """Closed form. lam>0 → ridge (L2), bias term (col 0) left unpenalized."""
    d = Xb.shape[1]
    R = lam * np.eye(d)
    R[0, 0] = 0.0
    return np.linalg.solve(Xb.T @ Xb + R, Xb.T @ t)


class LinearRegression:
    """method = 'normal' | 'gd' | 'ridge'. Returns coefficients with intercept_
    and coef_ exposed sklearn-style for validation."""

    def __init__(self, method="normal", lam=0.0, alpha=0.01, num_iters=5000):
        self.method = method
        self.lam = lam
        self.alpha = alpha
        self.num_iters = num_iters

    def fit(self, X, t):
        Xb = _add_bias(np.asarray(X, dtype=float))
        t = np.asarray(t, dtype=float)
        if self.method in ("normal", "ridge"):
            lam = self.lam if self.method == "ridge" else 0.0
            theta = normal_equation(Xb, t, lam=lam)
            self.cost_history_ = None
        elif self.method == "gd":
            theta = np.zeros(Xb.shape[1])
            hist = []
            n = len(t)
            for _ in range(self.num_iters):
                grad = (Xb.T @ (Xb @ theta - t)) / n
                if self.lam:                       # optional L2 (bias unpenalized)
                    reg = self.lam * theta; reg[0] = 0.0
                    grad = grad + reg
                theta = theta - self.alpha * grad
                hist.append(0.5 * np.mean((Xb @ theta - t) ** 2))
            self.cost_history_ = hist
        else:
            raise ValueError(self.method)
        self.theta_ = theta
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self

    def predict(self, X):
        return _add_bias(np.asarray(X, dtype=float)) @ self.theta_
