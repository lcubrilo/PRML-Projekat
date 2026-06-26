"""B1 + label: load mdabbert, drop result/leakage columns, drop weird outcomes,
build the joint 6-class label (winner x method).

Column facts verified in docs/DATASETS.md.
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
    return pd.read_csv(path)


def build_label(df: pd.DataFrame) -> pd.Series:
    mask = (
        df["Winner"].isin(["Red", "Blue"])
        & df["finish"].isin(METHOD_MAP)
    )

    method = df.loc[mask, "finish"].map(METHOD_MAP)
    return df.loc[mask, "Winner"] + "-" + method


def load_clean() -> tuple[pd.DataFrame, pd.Series]:
    df = load_raw()
    y = build_label(df)

    df_clean = df.loc[y.index].copy()
    X_pool = df_clean.drop(columns=LEAK_COLS, errors="ignore")

    return X_pool, y
