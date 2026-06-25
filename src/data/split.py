"""B4: chronological split + train-only scaling.

STUB - fill the TODOs.
"""
import pandas as pd


def chronological_split(X: pd.DataFrame, y: pd.Series, dates: pd.Series, test_frac: float = 0.2):
    """Most-recent `test_frac` fights = test set (train on the past, test on the future).
    TODO:
      - order = dates.sort_values().index ; n_test = int(len(X)*test_frac)
      - test = the last n_test by date, train = the rest
    Return X_train, X_test, y_train, y_test.
    """
    raise NotImplementedError


def scale_train_only(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Standardize using TRAIN statistics only (no leakage from test).
    TODO:
      mu, sd = X_train.mean(), X_train.std().replace(0, 1)
      return (X_train - mu)/sd, (X_test - mu)/sd
    (Only scale numeric columns; leave one-hot columns as-is if you prefer.)
    """
    raise NotImplementedError
