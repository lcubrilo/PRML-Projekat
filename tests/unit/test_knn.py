import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from src.baselines import KNNClassifier

def _split():
    X, y = load_iris(return_X_y=True)
    ys = np.array(["a","b","c"])[y]  # arbitrary string labels
    return train_test_split(X, ys, test_size=0.3, random_state=0, stratify=y)

def test_matches_sklearn():
    Xtr, Xte, ytr, yte = _split()
    for k in (1, 3, 5, 7, 15):
        ours = KNNClassifier(k=k).fit(Xtr, ytr).predict(Xte)
        ref = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr).predict(Xte)
        assert np.mean(ours == ref) > 0.95


if __name__ == '__main__':
    g = dict(globals())
    fns = [(n, f) for n, f in g.items() if n.startswith('test_') and callable(f)]
    for n, f in fns:
        f(); print('PASS', n)
    print(f'{len(fns)} unit checks passed')
