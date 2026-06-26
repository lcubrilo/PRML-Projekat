"""Validation for the betting-market benchmark helpers (src/data/odds.py).

Conversions are checked against hand-computed values; the de-vig + market
builders are checked against the real committed CSV (probabilities must sum to
1 on rows with full coverage). No sklearn here - this is pure arithmetic.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import pandas as pd

from src.data.odds import american_to_prob, devig, moneyline_market, method_market

CSV = os.path.join(os.path.dirname(__file__), '..', '..',
                   'data', 'raw', 'mdabbert', 'ufc-master.csv')


def test_american_to_prob_known_values():
    # +100/-100 -> 0.5; +150 -> 0.40; -150 -> 0.60; -200 -> 0.6667
    got = american_to_prob([100, -100, 150, -150, -200])
    exp = [0.5, 0.5, 0.40, 0.60, 200 / 300]
    assert np.allclose(got, exp)


def test_american_to_prob_nan_propagates():
    assert np.isnan(american_to_prob([np.nan]))[0]


def test_devig_removes_overround():
    # a -110/-110 two-way market: each implied ~0.5238, sum 1.0476 -> de-vig 0.5/0.5
    p = american_to_prob([-110, -110])
    out = devig(p[None, :])
    assert np.allclose(out, [[0.5, 0.5]])
    assert np.isclose(out.sum(), 1.0)


def test_devig_nan_row():
    out = devig(np.array([[0.4, np.nan]]))
    assert np.all(np.isnan(out))


def _df():
    return pd.read_csv(CSV)


def test_moneyline_market_sums_to_one():
    m = moneyline_market(_df()).dropna()
    assert len(m) > 1000                              # moneyline present on most fights
    assert np.allclose(m.sum(axis=1), 1.0)
    assert (m.values >= 0).all() and (m.values <= 1).all()


def test_method_market_six_classes_and_coverage():
    df = _df()
    m = method_market(df)
    assert list(m.columns) == list(method_market.__globals__['METHOD_ODDS_COLS'])
    full = m.dropna()
    assert np.allclose(full.sum(axis=1), 1.0)         # de-vigged distribution
    assert len(full) / len(m) > 0.6                   # ~82% coverage per DATASETS.md


def test_method_market_reindex_to_model_classes():
    df = _df()
    classes = ["Blue-DEC", "Blue-KO", "Blue-SUB", "Red-DEC", "Red-KO", "Red-SUB"]
    m = method_market(df, classes=classes)
    assert list(m.columns) == classes                 # lines up with model.classes_
    assert np.allclose(m.dropna().sum(axis=1), 1.0)


if __name__ == '__main__':
    g = dict(globals())
    fns = [(n, f) for n, f in g.items() if n.startswith('test_') and callable(f)]
    for n, f in fns:
        f(); print('PASS', n)
    print(f'{len(fns)} unit checks passed')
