"""B2 + B3: build the feature matrix and symmetrize corners.

make_features selects the pre-fight feature columns; symmetrize randomizes the
red/blue corners so the base rate is 50/50. Betting-market columns are excluded
from the features (they are benchmark-only - see MARKET_COLS).
"""
import numpy as np
import pandas as pd

# Betting-market columns (moneyline odds + expected value). These stay for the odds
# BENCHMARK only and must never be features, or the model just relearns the market.
MARKET_COLS = ["R_odds", "B_odds", "R_ev", "B_ev"]


def make_features(df: pd.DataFrame, use_diffs: bool = True, use_absolutes: bool = True) -> pd.DataFrame:
    """Select pre-fight features. The dataset already ships BOTH:
      - difference cols: name ends in '_dif'  (reach_dif, age_dif, ko_dif, sig_str_dif, ...)
      - absolute cols:  R_*/B_*               (R_Reach_cms, B_age, R_avg_SIG_STR_landed, ...)"""
    if not use_diffs and not use_absolutes:
        raise ValueError("At least one feature group must be selected.")

    parts = []

    if use_diffs:
        diff_cols = [c for c in df.columns if c.endswith("_dif")]
        parts.append(df[diff_cols].copy())
        
# TODO (feature engineering):
# Try alternative encoding for stance features:
# - current: separate one-hot encoding for Red and Blue stance
# - experiment with Red/Blue pair encoding (NxN combinations)
# - group rare stance categories if they hurt model performance (SAMME)

    if use_absolutes:
        abs_cols = [
            c for c in df.columns
            if (c.startswith("R_") or c.startswith("B_"))
            and c not in ["R_fighter", "B_fighter", "R_Stance", "B_Stance", *MARKET_COLS]
            and "odds" not in c.lower()  # drop any market/odds column (benchmark-only)
        ]
        parts.append(df[abs_cols].copy())

    cat_cols = [c for c in ["R_Stance", "B_Stance", "weight_class"] if c in df.columns]

    if cat_cols:
        cats = df[cat_cols].copy()
        cats = cats.fillna("Unknown")
        cats = pd.get_dummies(cats, columns=cat_cols)
        parts.append(cats)

    X = pd.concat(parts, axis=1)

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Missing numerical values, mostly debut statistics, are filled with median.
    numeric_cols = X.select_dtypes(include="number").columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    X = X.fillna(0)

    return X


def symmetrize(X: pd.DataFrame, y: pd.Series, seed: int = 0):
    """Randomly swap red/blue per fight so the base rate is 50/50 (kills the
    'just predict red' shortcut).
     IMPORTANT (issue #4): a swap must be CONSISTENT for the swapped rows:
      - flip the SIGN of every '*_dif' column, AND/OR swap R_* <-> B_* columns,
      - flip the label's winner side (Red-KO <-> Blue-KO; method stays).
    Swapping only the first two columns is wrong."""
    
    rng = np.random.default_rng(seed)

    X_sym = X.copy()
    y_sym = y.copy()

    swap_mask = rng.random(len(X_sym)) < 0.5
    swap_idx = X_sym.index[swap_mask]

    diff_cols = [c for c in X_sym.columns if c.endswith("_dif")]
    X_sym.loc[swap_idx, diff_cols] = -X_sym.loc[swap_idx, diff_cols]

    r_cols = [c for c in X_sym.columns if c.startswith("R_")]

    for r_col in r_cols:
        b_col = "B_" + r_col[2:]

        if b_col in X_sym.columns:
            temp = X_sym.loc[swap_idx, r_col].copy()
            X_sym.loc[swap_idx, r_col] = X_sym.loc[swap_idx, b_col]
            X_sym.loc[swap_idx, b_col] = temp

    def flip_label(label: str) -> str:
        side, method = label.split("-")

        if side == "Red":
            return "Blue-" + method

        return "Red-" + method

    y_sym.loc[swap_idx] = y_sym.loc[swap_idx].apply(flip_label)

    return X_sym, y_sym
