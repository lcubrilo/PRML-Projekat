"""B2 + B3: build the feature matrix and symmetrize corners.

STUB - fill the TODOs.
"""
import pandas as pd


def make_features(df: pd.DataFrame, use_diffs: bool = True, use_absolutes: bool = True) -> pd.DataFrame:
    """Select pre-fight features. The dataset already ships BOTH:
      - difference cols: name ends in '_dif'  (reach_dif, age_dif, ko_dif, sig_str_dif, ...)
      - absolute cols:  R_*/B_*               (R_Reach_cms, B_age, R_avg_SIG_STR_landed, ...)
    TODO:
      - pick columns according to the flags (diffs only / absolutes only / both),
      - one-hot encode categoricals: stance, weight_class,
      - handle debut NaNs (e.g. fill with 0/median, or drop debut rows - decide + document),
      - keep `date` out of the feature matrix but carry it for the split.
    NOTE (experiment, issue #3): diffs-only is smaller/faster but assumes linear, level-
    independent effects; diffs+absolutes lets the model learn interactions. Try both.
    """
    raise NotImplementedError


def symmetrize(X: pd.DataFrame, y: pd.Series, seed: int = 0):
    """Randomly swap red/blue per fight so the base rate is 50/50 (kills the
    'just predict red' shortcut).

    IMPORTANT (issue #4): a swap must be CONSISTENT for the swapped rows:
      - flip the SIGN of every '*_dif' column, AND/OR swap R_* <-> B_* columns,
      - flip the label's winner side (Red-KO <-> Blue-KO; method stays).
    Swapping only the first two columns is wrong.
    TODO: rng = np.random.default_rng(seed); pick a random ~50% subset of rows; apply.
    Return (X_sym, y_sym).
    """
    raise NotImplementedError
