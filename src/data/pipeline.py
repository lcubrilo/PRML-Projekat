"""Leakage-safe dataset pipeline (composition of the B1-B4 steps).

build_dataset wires the whole thing in the leakage-safe order:
    load -> make_features(impute=False) -> chronological split
         -> symmetrize corners -> train-only impute -> train-only scale

The model trains and is scored on the SYMMETRIZED split (so it cannot exploit the
'red usually wins' shortcut). It also returns an ORIGINAL-corner test set, transformed
with the SAME train statistics, used for the reference baselines that only make sense in
original-corner space: the always-red baseline and the betting-market benchmark.

Design note (flag for the team): main model metrics are reported on the symmetrized
test (50/50 base rate = pure skill); the always-red and market references use the
original-corner test. Both are derived from the same chronological split.
"""
from dataclasses import dataclass
import pandas as pd

from .load import load_clean
from .features import make_features, symmetrize
from .split import chronological_split, scale_train_only


def impute_train_only(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fill NaNs with TRAIN medians (then 0), applied to both train and test.

    The leakage-safe counterpart of make_features' global median fill: the medians
    come from train only, so no test information leaks into the imputation.
    """
    num = X_train.select_dtypes(include="number").columns
    med = X_train[num].median()
    Xtr, Xte = X_train.copy(), X_test.copy()
    Xtr[num] = Xtr[num].fillna(med)
    Xte[num] = Xte[num].fillna(med)
    return Xtr.fillna(0), Xte.fillna(0)


@dataclass
class Dataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame        # symmetrized test (the main evaluation set)
    y_test: pd.Series
    X_test_orig: pd.DataFrame   # original-corner test (for always-red / market refs)
    y_test_orig: pd.Series      # original-corner test labels
    df: pd.DataFrame            # raw rows (odds, original corners), full dataset
    test_index: pd.Index        # df index of the test fights


def build_dataset(test_frac: float = 0.2, seed: int = 0) -> Dataset:
    df, y = load_clean()
    X = make_features(df, impute=False)                 # defer imputation to train-only
    Xtr, Xte, ytr, yte = chronological_split(X, y, df["date"], test_frac)
    test_index = Xte.index

    # symmetrize corners on train and test (kills the 'always red' shortcut for the model)
    Xtr_s, ytr_s = symmetrize(Xtr, ytr, seed=seed)
    Xte_s, yte_s = symmetrize(Xte, yte, seed=seed + 1)

    # train-only impute (fit medians on symmetrized train, apply to each test set)
    Xtr_imp, Xte_imp = impute_train_only(Xtr_s, Xte_s)
    _,       Xte_o_imp = impute_train_only(Xtr_s, Xte)   # original-corner test, same train fit

    # train-only scale (fit on imputed train, apply to each test set)
    Xtr_sc, Xte_sc = scale_train_only(Xtr_imp, Xte_imp)
    _,      Xte_o_sc = scale_train_only(Xtr_imp, Xte_o_imp)

    return Dataset(
        X_train=Xtr_sc, y_train=ytr_s,
        X_test=Xte_sc, y_test=yte_s,
        X_test_orig=Xte_o_sc, y_test_orig=yte,
        df=df, test_index=test_index,
    )
