"""B4: chronological split + train-only scaling.

chronological_split keeps the most recent fights as the test set (train on the
past, test on the future). scale_train_only standardizes using train statistics
only, leaving one-hot/binary columns untouched.
"""
import pandas as pd


def chronological_split(X: pd.DataFrame, y: pd.Series, dates: pd.Series, test_frac: float = 0.2):
    """Most-recent `test_frac` fights = test set (train on the past, test on the future)."""
    dates = pd.to_datetime(dates)

    order = dates.loc[X.index].sort_values().index
    n_test = int(len(X) * test_frac)

    test_idx = order[-n_test:]
    train_idx = order[:-n_test]

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    return X_train, X_test, y_train, y_test


def scale_train_only(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Standardize using TRAIN statistics only (no leakage from test)."""
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    numeric_cols = X_train.select_dtypes(include="number").columns

    binary_cols = [
        c for c in numeric_cols
        if set(X_train[c].dropna().unique()).issubset({0, 1})
    ]

    scale_cols = [c for c in numeric_cols if c not in binary_cols]

    mu = X_train[scale_cols].mean()
    sd = X_train[scale_cols].std().replace(0, 1)

    X_train_scaled[scale_cols] = (X_train[scale_cols] - mu) / sd
    X_test_scaled[scale_cols] = (X_test_scaled[scale_cols] - mu) / sd

    return X_train_scaled, X_test_scaled
