"""Naive reference classifiers (the floor any real model has to beat).

- MajorityClassifier: always predicts the most frequent class in the training set
  (e.g. always "decision" for the joint winner x method target).
- ConstantClassifier: always predicts one fixed label given up front (e.g. an
  always-pick-a-given-corner rule).

Both follow the same sklearn-style fit/predict/predict_proba/score API as the rest
of src/baselines so they drop straight into the evaluation code.

IMPORTANT - the "always-red" baseline must be measured on the ORIGINAL corners,
i.e. BEFORE B3 symmetrization. Symmetrization randomizes red/blue to a 50/50 base
rate on purpose, so "always pick red" scores ~50% there and the ~58-64% red-corner
edge (the thing this baseline is meant to capture) disappears. So evaluate it on the
un-symmetrized data, and as a WINNER-level reference: collapse both the prediction
("Red") and the 6-class truth to the winner side (Red/Blue) and score that. The
real model is still trained/evaluated on the symmetrized data; only this naive
reference uses the original corners.
"""
import numpy as np


class MajorityClassifier:
    """Predict the training set's most frequent class for everything."""

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.majority_ = self.classes_[np.argmax(counts)]
        self.class_freq_ = counts / counts.sum()  # used as a (constant) predict_proba
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_)

    def predict_proba(self, X):
        return np.tile(self.class_freq_, (len(X), 1))

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


class ConstantClassifier:
    """Always predict `constant`, which must appear in the training labels."""

    def __init__(self, constant):
        self.constant = constant

    def fit(self, X, y):
        self.classes_ = np.unique(np.asarray(y))
        if self.constant not in self.classes_:
            raise ValueError(f"constant {self.constant!r} is not among the training labels")
        return self

    def predict(self, X):
        return np.full(len(X), self.constant)

    def predict_proba(self, X):
        proba = (self.classes_ == self.constant).astype(float)
        return np.tile(proba, (len(X), 1))

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
