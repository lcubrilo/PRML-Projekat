"""B1 + label: load mdabbert, drop result/leakage columns, drop weird outcomes,
build the joint 6-class label (winner x method).

STUB - fill the TODOs. Column facts already verified (see docs/DATASETS.md).
"""
from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "mdabbert" / "ufc-master.csv"

# Columns that describe the RESULT of the fight -> never use as features (leakage).
# total_fight_time_secs is the sneaky one: it is THIS fight's duration.
LEAK_COLS = ["Winner", "finish", "finish_details", "finish_round",
             "finish_round_time", "total_fight_time_secs", "empty_arena"]

# finish -> 3 method buckets. Anything not here is a "weird outcome" we DROP
# (DQ, Overturned, CNC, draws, and rows with unknown method).
METHOD_MAP = {"KO/TKO": "KO", "SUB": "SUB", "U-DEC": "DEC", "S-DEC": "DEC", "M-DEC": "DEC"}


def load_raw(path: Path = RAW) -> pd.DataFrame:
    """Read the CSV.  TODO: return pd.read_csv(path)."""
    raise NotImplementedError


def build_label(df: pd.DataFrame) -> pd.Series:
    """Joint 6-class label, e.g. 'Red-KO', 'Blue-DEC'.
    TODO:
      1. mask = df.Winner.isin(['Red','Blue']) & df.finish.isin(METHOD_MAP)
      2. method = df.loc[mask,'finish'].map(METHOD_MAP)
      3. return df.loc[mask,'Winner'] + '-' + method
    Weird outcomes drop out automatically (they fail the mask).
    """
    raise NotImplementedError


def load_clean() -> tuple[pd.DataFrame, pd.Series]:
    """Return (df_features_pool, label) on the SAME (filtered) index.
    TODO: load_raw -> build_label -> align rows -> drop LEAK_COLS from the feature pool.
    Keep `date` (needed for the chronological split) and identifiers for EDA.
    """
    raise NotImplementedError
