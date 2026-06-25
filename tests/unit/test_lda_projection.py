import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np

from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA
from src.baselines import LDAProjection

def test_top_direction_aligns_with_sklearn():
    X, y = load_iris(return_X_y=True)
    W = LDAProjection(n_components=1).fit(X, y).components_[:, 0]
    ref = skLDA().fit(X, y).scalings_[:, 0]
    cos = abs(np.dot(W, ref) / (np.linalg.norm(W) * np.linalg.norm(ref)))
    assert cos > 0.99


if __name__ == '__main__':
    g = dict(globals())
    fns = [(n, f) for n, f in g.items() if n.startswith('test_') and callable(f)]
    for n, f in fns:
        f(); print('PASS', n)
    print(f'{len(fns)} unit checks passed')
