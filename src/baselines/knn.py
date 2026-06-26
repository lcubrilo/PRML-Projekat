"""k-Nearest Neighbors - ported from `course_materials/08_knn/8_kNN_solution.ipynb`.

The core functions (`euclidean_distance`, `manhattan_distance`, `knn_predict_point`,
`knn_predict`) are the course implementations, kept faithful. `KNNClassifier` is a
thin sklearn-style wrapper added for the project so it accepts arbitrary labels
(e.g. 'KO/TKO', 'Submission', 'Decision') and integrates with our evaluation code.

This is a *primary* implementation. sklearn appears only in tests (validation).
"""
import numpy as np


# --- course implementations (faithful) -------------------------------------
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))


def knn_predict_point(X_train, y_train, x_query, k=3, distance_fn=euclidean_distance):
    """Majority vote over the k nearest neighbours. Expects y_train as
    non-negative integer labels (0..C-1). Ties broken toward the lowest label
    index (np.argmax)."""
    distances = np.array([distance_fn(x, x_query) for x in X_train])
    nn_idx = np.argsort(distances)[:k]
    nn_labels = y_train[nn_idx]
    counts = np.bincount(nn_labels)
    return np.argmax(counts)


def knn_predict(X_train, y_train, X_test, k=3, distance_fn=euclidean_distance):
    return np.array(
        [knn_predict_point(X_train, y_train, x, k=k, distance_fn=distance_fn) for x in X_test]
    )


# --- project wrapper -------------------------------------------------------
class KNNClassifier:
    """sklearn-style wrapper around the course kNN. Handles arbitrary (e.g.
    string) class labels by encoding them to 0..C-1 internally."""

    def __init__(self, k=3, distance_fn=euclidean_distance):
        self.k = k
        self.distance_fn = distance_fn

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y_enc = np.unique(np.asarray(y), return_inverse=True)
        self._X = X
        self._y = y_enc.astype(int)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        enc = knn_predict(self._X, self._y, X, k=self.k, distance_fn=self.distance_fn)
        return self.classes_[enc]

    def predict_proba(self, X):
        """Class probabilities = fraction of the k nearest neighbours in each class."""
        X = np.asarray(X, dtype=float)
        K = len(self.classes_)
        out = np.zeros((len(X), K))
        for i, x in enumerate(X):
            d = np.array([self.distance_fn(xt, x) for xt in self._X])
            nn = np.argsort(d)[:self.k]
            out[i] = np.bincount(self._y[nn], minlength=K) / self.k
        return out

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
