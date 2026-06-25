import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from src.baselines import kde_estimate, ParzenClassifier

def test_kde_matches_sklearn():
    rng = np.random.default_rng(0); s = rng.normal(0, 1, 500); xg = np.linspace(-3, 3, 50)
    ours = kde_estimate(xg, s, h=0.4)
    skl = np.exp(KernelDensity(bandwidth=0.4, kernel="gaussian").fit(s.reshape(-1,1)).score_samples(xg.reshape(-1,1)))
    assert np.allclose(ours, skl, atol=1e-6)

def test_parzen_classifier_iris():
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    assert ParzenClassifier(h=0.5).fit(Xtr, ytr).score(Xte, yte) > 0.9


if __name__ == '__main__':
    g = dict(globals())
    fns = [(n, f) for n, f in g.items() if n.startswith('test_') and callable(f)]
    for n, f in fns:
        f(); print('PASS', n)
    print(f'{len(fns)} unit checks passed')
