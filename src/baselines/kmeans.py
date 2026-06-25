"""k-means - ported from `11_clustering/11_clustering_solution.ipynb` (faithful),
plus a thin sklearn-style wrapper. The KMeans++ extension (if chosen) will
subclass/replace `initialize_centroids` only.

Primary implementation; sklearn used only in tests.
"""
import numpy as np
from .knn import euclidean_distance


def initialize_centroids(X, k, seed=None):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=k, replace=False)
    return X[idx].copy()


def assign_clusters(X, centroids):
    labels = []
    for x in X:
        dists = [euclidean_distance(x, c) for c in centroids]
        labels.append(np.argmin(dists))
    return np.array(labels)


def recompute_centroids(X, labels, k):
    centroids = []
    for j in range(k):
        cluster_points = X[labels == j]
        if len(cluster_points) == 0:
            centroids.append(np.zeros(X.shape[1]))
        else:
            centroids.append(np.mean(cluster_points, axis=0))
    return np.array(centroids)


def kmeans(X, k, max_iters=50, seed=None):
    centroids = initialize_centroids(X, k, seed=seed)
    history = [centroids.copy()]
    labels = assign_clusters(X, centroids)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = recompute_centroids(X, labels, k)
        history.append(new_centroids.copy())
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids, labels, history


def kmeans_objective(X, labels, centroids):
    """Within-cluster sum of squares (a.k.a. inertia)."""
    total = 0.0
    for j in range(len(centroids)):
        cluster_points = X[labels == j]
        total += np.sum((cluster_points - centroids[j]) ** 2)
    return total


class KMeans:
    """sklearn-style wrapper. `n_init` restarts keep the lowest-inertia run
    (vanilla k-means is seed-sensitive - this is what k-means++ improves on)."""

    def __init__(self, k=3, max_iters=50, n_init=10, seed=0):
        self.k = k
        self.max_iters = max_iters
        self.n_init = n_init
        self.seed = seed

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        best = None
        for i in range(self.n_init):
            c, lab, _ = kmeans(X, self.k, self.max_iters, seed=self.seed + i)
            obj = kmeans_objective(X, lab, c)
            if best is None or obj < best[0]:
                best = (obj, c, lab)
        self.inertia_, self.cluster_centers_, self.labels_ = best
        return self

    def predict(self, X):
        return assign_clusters(np.asarray(X, dtype=float), self.cluster_centers_)
