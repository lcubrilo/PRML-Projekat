"""D4: betting-market benchmark helpers.

Convert the sportsbook odds shipped in `mdabbert/ufc-master.csv` into de-vigged
probabilities so our from-scratch model can be scored against the market by
log-loss / Brier. The market is a *rival forecaster* here - odds are NEVER used
as model features (that would just relearn the market; see CLAUDE.md Section 6).

Two markets are available:
  * moneyline (`R_odds` / `B_odds`)          -> P(Red win), P(Blue win)
  * per-method props (`r_ko_odds` ... )       -> joint distribution over the
    6 winner x method classes, directly comparable to the model's predict_proba.
    ~80% of fights carry full prop coverage; the rest come back as NaN rows.
"""
import numpy as np
import pandas as pd

# 6-class label -> its per-method prop column. Labels match src/data/load.py
# build_label: f"{Winner}-{method}" with method in {KO, SUB, DEC}.
METHOD_ODDS_COLS = {
    "Red-KO":  "r_ko_odds",  "Red-SUB":  "r_sub_odds",  "Red-DEC":  "r_dec_odds",
    "Blue-KO": "b_ko_odds",  "Blue-SUB": "b_sub_odds",  "Blue-DEC": "b_dec_odds",
}


def american_to_prob(odds) -> np.ndarray:
    """American moneyline odds -> implied probability (still vigged).

    +150 -> 0.40, -150 -> 0.60, +100/-100 -> 0.50. NaNs propagate.
    """
    odds = np.asarray(odds, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.where(odds > 0, 100.0 / (odds + 100.0), -odds / (-odds + 100.0))


def devig(prob_matrix) -> np.ndarray:
    """Normalise each row of implied probabilities to sum to 1.

    Removes the bookmaker overround (the implied probs of a market sum to >1;
    the excess is the vig). A row with any NaN entry -> all-NaN (the market is
    incomplete for that fight, so it has no usable benchmark).
    """
    P = np.asarray(prob_matrix, dtype=float)
    return P / P.sum(axis=1, keepdims=True)


def moneyline_market(df: pd.DataFrame) -> pd.DataFrame:
    """De-vigged P(Red win), P(Blue win) from `R_odds` / `B_odds`.

    Returns a DataFrame indexed like `df` with columns ['Red', 'Blue'].
    """
    P = devig(np.column_stack([american_to_prob(df["R_odds"]),
                               american_to_prob(df["B_odds"])]))
    return pd.DataFrame(P, index=df.index, columns=["Red", "Blue"])


def method_market(df: pd.DataFrame, classes=None) -> pd.DataFrame:
    """De-vigged market distribution over the 6 winner x method classes.

    Built from the per-method prop odds and de-vigged across all six outcomes,
    so it lines up 1:1 with a 6-class model's `predict_proba`. Fights missing any
    prop come back as NaN rows (drop them before scoring).

    Parameters
    ----------
    classes : optional sequence of class labels. If given, columns are reindexed
        to match (e.g. pass `model.classes_` so the market lines up with the
        model's column order before computing log-loss).
    """
    labels = list(METHOD_ODDS_COLS)
    P = devig(np.column_stack([american_to_prob(df[METHOD_ODDS_COLS[c]]) for c in labels]))
    out = pd.DataFrame(P, index=df.index, columns=labels)
    return out.reindex(columns=list(classes)) if classes is not None else out


def market_benchmark(y_true, model_proba, market_proba, classes) -> dict:
    """Score our model against the de-vigged market on the SAME fights.

    Both forecasters output a probability over `classes`, so we compare them by
    log-loss and Brier (the proper scoring rules for probabilistic forecasts).
    Only fights with full market coverage are used: rows where `market_proba` has
    any NaN are dropped from BOTH sides so the comparison is on identical fights.

    Parameters
    ----------
    y_true : array of true labels, length n.
    model_proba : (n, K) array, columns aligned to `classes` (model.predict_proba).
    market_proba : (n, K) array or DataFrame from `method_market(df, classes=...)`.
    classes : the class order both probability matrices use.

    Returns
    -------
    dict with 'log_loss' and 'brier' sub-dicts, each holding 'model' and 'market',
    plus 'n_fights' actually scored.
    """
    from src.metrics import log_loss, brier_score

    y_true = np.asarray(y_true)
    model_proba = np.asarray(model_proba, dtype=float)
    market_proba = np.asarray(market_proba, dtype=float)

    covered = ~np.isnan(market_proba).any(axis=1)
    yt = y_true[covered]
    m_model = model_proba[covered]
    m_market = market_proba[covered]
    return {
        "log_loss": {"model": log_loss(yt, m_model, classes=classes),
                     "market": log_loss(yt, m_market, classes=classes)},
        "brier": {"model": brier_score(yt, m_model, classes=classes),
                  "market": brier_score(yt, m_market, classes=classes)},
        "n_fights": int(covered.sum()),
    }
