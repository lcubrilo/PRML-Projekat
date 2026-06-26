"""Generative Gaussian classifiers - Naive Bayes / LDA / QDA.

Generalizes the course's 2-class GDA (`05_quadratic_classifiers`: `predict_gda`,
which scores each class by N(x; mu_c, Sigma_c)*prior_c and takes the argmax) to
the multiclass, multivariate setting needed for this project (e.g. method-of-
victory: 3 classes). Same generative recipe as the course, just:
  - QDA: a separate covariance per class,
  - LDA: one shared pooled covariance (course `Sigma_shared`),
  - NB : diagonal covariance (independent features).

Primary implementation; sklearn used only in tests.
"""
import numpy as np


def _log_gaussian(X, mu, Sigma, reg):
    """Multivariate log N(x; mu, Sigma) for each row of X, with a small ridge
    on Sigma for numerical stability (correlated UFC features ⇒ near-singular)."""
    d = X.shape[1]
    Sigma = Sigma + reg * np.eye(d)
    diff = X - mu
    sign, logdet = np.linalg.slogdet(Sigma)
    sol = np.linalg.solve(Sigma, diff.T).T          # Sigma^-1 (x-mu)
    maha = np.einsum("ij,ij->i", diff, sol)         # quadratic form
    return -0.5 * (d * np.log(2 * np.pi) + logdet + maha)


class GaussianClassifier:
    """kind = 'qda' (per-class cov) | 'lda' (shared cov) | 'nb' (diagonal cov).
    Labels may be arbitrary (encoded internally). Covariances are MLE estimates
    (divide by class count / N), matching the course."""

    def __init__(self, kind="qda", reg=1e-6):
        assert kind in ("qda", "lda", "nb")
        self.kind = kind
        self.reg = reg

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n, d = X.shape
        self.means_, self.priors_, self.covs_ = [], [], []
        pooled = np.zeros((d, d))
        for c in self.classes_:
            Xc = X[y == c]
            mu = Xc.mean(axis=0)
            diff = Xc - mu
            cov = (diff.T @ diff) / len(Xc)             # MLE (÷ n_c)
            if self.kind == "nb":
                cov = np.diag(np.diag(cov))
            self.means_.append(mu)
            self.priors_.append(len(Xc) / n)
            self.covs_.append(cov)
            pooled += diff.T @ diff
        if self.kind == "lda":
            shared = pooled / n                         # course Sigma_shared (÷ N)
            self.covs_ = [shared for _ in self.classes_]
        return self

    def predict_log_proba(self, X):
        X = np.asarray(X, dtype=float)
        scores = np.column_stack([
            _log_gaussian(X, mu, cov, self.reg) + np.log(prior)
            for mu, cov, prior in zip(self.means_, self.covs_, self.priors_)
        ])
        return scores

    def predict_proba(self, X):
        """Posterior probabilities: softmax over the (unnormalized) joint log-scores."""
        log = self.predict_log_proba(X)
        log = log - log.max(axis=1, keepdims=True)   # numerical stability
        p = np.exp(log)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
