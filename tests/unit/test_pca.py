import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as skPCA
from src.baselines import PCA

def test_explained_variance_matches():
    X, _ = load_iris(return_X_y=True)
    assert np.allclose(PCA().fit(X).explained_variance_ratio_,
                       skPCA().fit(X).explained_variance_ratio_, atol=1e-6)

def test_inverse_transform_reconstructs():
    X, _ = load_iris(return_X_y=True)
    p = PCA().fit(X)
    assert np.allclose(p.inverse_transform(p.transform(X)), X, atol=1e-8)


if __name__ == '__main__':
    g = dict(globals())
    fns = [(n, f) for n, f in g.items() if n.startswith('test_') and callable(f)]
    for n, f in fns:
        f(); print('PASS', n)
    print(f'{len(fns)} unit checks passed')
