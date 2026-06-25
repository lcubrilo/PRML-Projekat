import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as skLinReg
from src.baselines import LinearRegression

def _data():
    return make_regression(n_samples=200, n_features=5, noise=8.0, random_state=0)

def test_normal_equation_matches_sklearn():
    X, t = _data(); ours = LinearRegression(method="normal").fit(X, t); ref = skLinReg().fit(X, t)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-6) and np.isclose(ours.intercept_, ref.intercept_, atol=1e-6)

def test_gd_converges_to_closed_form():
    X, t = _data(); ref = skLinReg().fit(X, t)
    ours = LinearRegression(method="gd", alpha=0.05, num_iters=20000).fit(X, t)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-2)


if __name__ == '__main__':
    g = dict(globals())
    fns = [(n, f) for n, f in g.items() if n.startswith('test_') and callable(f)]
    for n, f in fns:
        f(); print('PASS', n)
    print(f'{len(fns)} unit checks passed')
