# Dataset Inspection Findings

Findings from actually downloading/inspecting the data (Claude task). Status: ✅ verified locally · 📄 verified from source code/docs (not the CSV) · ⬜ unverified.

Local copies live in `data/raw/` (gitignored). Re-fetch: see commands at the bottom.

## Summary table

| Dataset | Access | Rows / cols | Pre-fight-safe features? | Odds? | Method+round target? | Names? | Best for |
|---|---|---|---|---|---|---|---|
| **komaksym** `stats_raw.csv` | ✅ GitHub raw, no auth | 8591 × 60 | ❌ **in-fight stats only** (must self-aggregate) | ❌ | ✅ `method`, `round` | ✅ | method/round targets; needs feature engineering |
| **komaksym** `stats_processed.csv` | ✅ GitHub | 8591 × ~50 | ❌ `delta_*` are **winner−loser** → leaks winner | ❌ | ✅ | winner/loser | NOT for winner prediction as-is |
| **komaksym** `raw_fighter_details.csv` | ✅ GitHub | bios | career aggregates (current, not as-of-fight) | ❌ | — | ✅ | fighter bio join (Height/Reach/Stance/DOB) |
| **komaksym** scorecards / merged | ✅ GitHub | — | — | ❌ | ✅ + judge scores | ✅ | **only source with judges' scorecards** (JudgeAI route) |
| **rajeevw/ufcdata** (Warrier) | ✅ Kaggle | `data.csv` 6012×144 | ✅ **pre-fight aggregates** (`*_avg_*`); leakage check passed | ❌ | target via `raw_total_fight_data.csv` `win_by` (join) | ✅ + `weight_class` | **leakage-safe features** (no odds) |
| **mdabbert/ultimate-ufc** | ✅ Kaggle | 7177×118, 2010–2026 | ✅ pre-fight aggregates (`*_avg_*`) | ✅ **moneyline 97% + per-method (KO/sub/dec) ~82%** | ✅ `finish`, `finish_round` | ✅ | **the one-file winner. method target + features + odds** |

## Key facts (decision-relevant)

**Columns we HAVE** (✅ confirmed in komaksym; consistent with Warrier code): fighter **names** + nicknames, **event date/location**, **weight class** (`bout_type`, e.g. "Welterweight Bout", "... Title Bout"), **method** of victory, **round** + time, referee, per-fighter **fight stats** (KD, sig strikes by target/position, TD, sub attempts, control time, reversals).

**Columns we DON'T have** (anywhere in the ufcstats lineage): **nationality, martial-art base, gym/team, coach.** ⇒ any nationality/style-origin angle needs **Tapology/Sherdog scraping** — confirmed out-of-scope-unless-pursued.

**Method-of-victory target (from komaksym `method`, n≈8.5k):**
- Decision (Unanimous 3098 + Split 818 + Majority 100) ≈ **4016 (47%)**
- KO/TKO 2703 (+ Doctor's stoppage 97) ≈ **2800 (33%)**
- Submission **1659 (19%)**
- Drop: Overturned 59, Could Not Continue 32, DQ 23, No-contest 90, Draw 63.
- ⇒ a clean 3-class problem, **moderate** imbalance (report per-class F1 + confusion matrix; consider stratification). ✅ viable.

**Red-corner skew:** komaksym `fight_outcome` = red_win 5417 / blue_win 3021 ≈ **64% red** — confirms the confound is present and strong.

**Leakage status by source:**
- Warrier/rajeevw **preprocessed** = the safe one: pre-fight career aggregates, and `preprocess.py` explicitly drops current-aggregate fighter columns ("information not yet occurred"). 📄
- komaksym `stats_raw` = **in-fight** per-bout stats → must build your own pre-fight rolling aggregates (the standard leakage-safe move). ✅ confirmed.
- komaksym `stats_processed` = `delta_*` features computed as **winner − loser** → trivially leaks the winner; only usable if re-symmetrized. ✅ confirmed.

**Oversampling claim — CLOSED (verified):** `preprocessed_data.csv` (5902×160) has **0 exact duplicate rows** → oversampling is not in the shipped file; it's a modeling-notebook step (`sklearn.utils.resample` in Warrier's `xgboost-Oversampled.ipynb`). No dedup needed unless you oversample yourself.

**Weird outcomes (mdabbert `finish`):** tiny — DQ 18 + CNC 3 + Overturned 3 (=24), plus 238 rows with a known winner but missing method, plus Draw 5 / No Contest 3. **Handling:** drop them (none is a learnable "method"). Leaves **6911 clean rows** for winner×method → 3-class method = DEC 3428 · KO/TKO 2223 · SUB 1260. For a winner-only model you can keep the 238 missing-method rows.

**LEAKAGE — important nuance (don't over-generalize):** only the **pre-aggregated** files (rajeevw `data.csv`/`preprocessed_data.csv`, mdabbert) are leakage-safe out of the box. komaksym `stats_raw` is **in-fight** (leaks if used directly — must self-aggregate); komaksym `stats_processed` **leaks the winner** (winner−loser deltas). So "no leakage" is true for what we'd actually use, not for every file.

**mdabbert leakage audit (verified, 118 cols) — where it CAN bite:** the *aggregate features don't leak* (debut-NaN check passed), but the FILE mixes in result columns you must EXCLUDE from features:
- **Targets/results:** `Winner`, `finish`, `finish_details`, `finish_round`, `finish_round_time`.
- **⚠️ The sneaky one — `total_fight_time_secs`:** this is *this fight's* duration (KO/TKO mean 363s vs decisions ~947s) → a result. Using it as a feature leaks the method massively. EXCLUDE.
- **Safe (pre-fight):** `*_avg_*`, `*_current_*_streak`, career `*_win_by_*` counts, the **pre-built `*_dif` difference features** (`ko_dif`, `sub_dif`, `sig_str_dif`, `win_streak_dif`…), physical (height/reach/age/stance), `R_odds`/`B_odds` + per-method props, rank, `no_of_rounds` (scheduled), gender, weight class.
- **`country` = EVENT location, NOT fighter nationality** (Seattle event → "USA"). So nationality is still absent → scraping needed.

⇒ "leakage-safe pipeline" for mdabbert is mostly **column selection** (drop the results above) + the usual preprocessing-leakage hygiene (scale/encode fit on train only). The historical-aggregate leakage is already handled by the dataset.

## Implications for the project decision (VERIFIED)
- **`mdabbert/ufc-master.csv` is the strongest single file:** it carries the `finish` target (KO/TKO 2223 · SUB 1260 · U/S/M-DEC 3432), pre-fight aggregate features (`*_avg_*`), **and** odds — both moneyline (`R_odds`/`B_odds`, 97%) and **per-method props** (`r_ko_odds`/`r_sub_odds`/`r_dec_odds` + blue, ~82% coverage). 2010–2026, 7177 fights.
- **CORRECTION — method-of-victory CAN be benchmarked vs the market.** Earlier I said odds were moneyline-only; **false**. mdabbert ships per-method prop odds, so a (winner×method) or finish-type prediction can be compared to a *market* forecaster, not just a majority-class baseline. This removes the main objection to the method-of-victory direction.
- **rajeevw** is the clean leakage-safe alternative if you don't need odds: `data.csv` (6012×144) has pre-fight aggregates + `Winner` + `weight_class`; the current-fight method (`win_by`) is in `raw_total_fight_data.csv` (same 6012 rows) → join for a method target. Leakage check passed: 1427/6012 rows have NaN priors (debutants) ⇒ aggregates are genuinely pre-fight.
- **JudgeAI/scorecard** route is uniquely enabled by komaksym (`SCORECARDS.csv`, merged file).

## Resolved (was ⬜, now ✅)
- mdabbert odds columns — confirmed (10 odds cols incl. per-method props).
- rajeevw method target — `win_by` lives in `raw_total_fight_data.csv`; `data.csv` has it only as *pre-fight career counts* (`R_win_by_*`), which are safe features, not the label.
- rajeevw leakage-safety — confirmed empirically (debut NaN priors).
- mdabbert `finish` 238 NaN + komaksym in-fight caveat still apply; clean by dropping NaN/`Overturned`/`DQ`/`CNC`.

## Re-fetch commands
```bash
# komaksym (no auth):
base="https://raw.githubusercontent.com/komaksym/UFC-DataLab/main/data"
curl -sL "$base/stats/stats_raw.csv"            -o data/raw/komaksym/stats_raw.csv
curl -sL "$base/stats/stats_processed.csv"      -o data/raw/komaksym/stats_processed.csv
curl -sL "$base/external_data/raw_fighter_details.csv" -o data/raw/komaksym/raw_fighter_details.csv
# Kaggle (token at ~/.kaggle/access_token, KGAT_ format; pip install kaggle):
python -m kaggle datasets download -d rajeevw/ufcdata           -p data/raw/rajeevw  --unzip
python -m kaggle datasets download -d mdabbert/ultimate-ufc-dataset -p data/raw/mdabbert --unzip
```
Downloaded files: `rajeevw/{data.csv, preprocessed_data.csv, raw_total_fight_data.csv, raw_fighter_details.csv}`, `mdabbert/{ufc-master.csv, upcoming.csv}`, `komaksym/{stats_raw, stats_processed, raw_fighter_details, ...}`.
