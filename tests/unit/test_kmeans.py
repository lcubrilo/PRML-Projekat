import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as skKMeans
from sklearn.metrics import adjusted_rand_score
from src.baselines import KMeans

def test_partition_matches_sklearn():
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
    ours = KMeans(k=4, seed=0).fit(X)
    ref = skKMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
    assert adjusted_rand_score(ours.labels_, ref.labels_) > 0.99


if __name__ == '__main__':
    g = dict(globals())
    fns = [(n, f) for n, f in g.items() if n.startswith('test_') and callable(f)]
    for n, f in fns:
        f(); print('PASS', n)
    print(f'{len(fns)} unit checks passed')
