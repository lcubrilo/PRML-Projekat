"""Leakage-safe dataset pipeline (composition of the B1-B4 steps).

build_dataset wires the whole thing in the leakage-safe order:
    load -> make_features(impute=False) -> chronological split
         -> symmetrize TRAIN corners -> train-only impute -> train-only scale

Symmetrization is a TRAIN-only device: it randomizes red/blue so the model cannot
learn the 'always pick red' shortcut. The TEST set keeps its real (original) corners,
so the model, the naive always-red baseline, and the betting market are all scored on
the same fights and are directly comparable. (We considered also scoring on a
symmetrized test for a 50/50-base 'pure skill' number, but it adds confusion for little
gain - see the report's discussion of alternatives.)
"""
from dataclasses import dataclass
import pandas as pd

from .load import load_clean
from .features import make_features, symmetrize
from .split import chronological_split, scale_train_only


def impute_train_only(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fill NaNs with TRAIN medians (then 0), applied to both train and test.

    The leakage-safe counterpart of make_features' global median fill: medians come
    from train only, so no test information leaks into the imputation.
    """
    num = X_train.select_dtypes(include="number").columns
    med = X_train[num].median()
    Xtr, Xte = X_train.copy(), X_test.copy()
    Xtr[num] = Xtr[num].fillna(med)
    Xte[num] = Xte[num].fillna(med)
    return Xtr.fillna(0), Xte.fillna(0)


@dataclass
class Dataset:
    X_train: pd.DataFrame   # symmetrized train (imputed + scaled)
    y_train: pd.Series
    X_test: pd.DataFrame    # original-corner test (imputed/scaled with TRAIN stats)
    y_test: pd.Series       # original-corner test labels
    df: pd.DataFrame        # raw rows (odds, original corners), full dataset
    test_index: pd.Index    # df index of the test fights


def build_dataset(test_frac: float = 0.2, seed: int = 0) -> Dataset:
    df, y = load_clean()
    X = make_features(df, impute=False)                 # defer imputation to train-only
    Xtr, Xte, ytr, yte = chronological_split(X, y, df["date"], test_frac)

    # symmetrize TRAIN corners only (kills the 'always red' shortcut); test stays original
    Xtr_s, ytr_s = symmetrize(Xtr, ytr, seed=seed)

    # train-only impute then train-only scale (fit on symmetrized train, apply to test)
    Xtr_imp, Xte_imp = impute_train_only(Xtr_s, Xte)
    Xtr_sc, Xte_sc = scale_train_only(Xtr_imp, Xte_imp)

    return Dataset(X_train=Xtr_sc, y_train=ytr_s, X_test=Xte_sc, y_test=yte,
                   df=df, test_index=Xte.index)
