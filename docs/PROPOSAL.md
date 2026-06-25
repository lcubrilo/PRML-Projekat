# Proposal — Decision Tracker

The TA proposal needs six points locked (see [CLAUDE.md §1](../CLAUDE.md)). This file tracks **what's decided vs. open**. Reasoning/menus live in [CLAUDE.md](../CLAUDE.md).

> Status: ✅ decided · 🟡 leaning · ⬜ open. **The team's submission to the TA may be a narrowed set, not a single pick** — e.g. "we lean X, would Y or Z be better?" is a valid proposal.

## Decision order & dependencies (decide top-down)

The TA's six points aren't in decision order. The actual dependency chain:

1. **Inspect the data first** (what columns even exist) — gates everything; a formulation is impossible if its target/columns aren't there. *(Claude task)*
2. **Problem formulation (b)** — *binds everything below.* Choose task type.
3. **Approach framing** — one of the TA's five (likely "baselines vs modern extensions").
4. **Dataset file (a)** — constrained by (b) + the odds decision (`mdabbert` if odds, else `rajeevw`).
5. **Baselines (c)** — determined by task type.
6. **Extension (d)** — task type + difficulty appetite.
7. **Reference (e)** → follows (d). **Plan (f)** → follows all.

## Six-point checklist

| # | Point | Status | Current |
|---|---|---|---|
| a | Dataset(s) | 🟡 | **`mdabbert/ufc-master.csv`** (verified: target + pre-fight features + odds). |
| b | Problem formulation | 🟡 | **Method of victory + winner** — joint (6-class) **or** cascade (open). |
| c | Baselines | ✅ | Bayes, LDA, QDA, kNN (+ linear floor) — implemented & validated (`src/baselines/`). |
| d | Extension | 🟡 | **SAMME-AdaBoost** leaning (RF not ruled out). |
| e | Reference | 🟡 | SAMME: Zhu, Zou, Rosset & Hastie (2009); base AdaBoost: Freund & Schapire (1997). |
| f | Plan | 🟡 | `src/data` pipeline → baselines → extension → eval + odds benchmark ([CLAUDE.md §7](../CLAUDE.md)). |

**Standing decision — odds usage: ✅ benchmark only** (not a feature, not unused). See the tension note in Direction 3.

## Leaning direction

### Direction 3 — Method of victory (multiclass) 🟡 *current lean*
- **Problem:** predict KO/TKO vs submission vs decision from pre-fight features.
- **Baselines:** Bayes, LDA, QDA, kNN.
- **Why:** more learnable from style features than winner; meaningful confusion matrix; rarely done publicly → distinctive.
- **Dataset (a): `mdabbert/ufc-master.csv`** — one file with `finish` target + pre-fight aggregates + odds. Verified (see [DATASETS.md](DATASETS.md)).
- **Odds benchmark NOW AVAILABLE for this direction (resolved).** mdabbert ships per-method props (`r_ko_odds/r_sub_odds/r_dec_odds` +blue, ~82%), so method-of-victory **can** be benchmarked vs the market — the earlier objection is gone. This makes Direction 3 dominate Direction 1 on the odds axis while keeping the richer target.
- **One caveat left — extension for multiclass:** Random Forest is natively multiclass (clean). Classic **AdaBoost is binary** → needs **SAMME** (Zhu et al. 2009) or one-vs-rest. GMM-via-EM is a generative multiclass option. → pick among RF / SAMME-AdaBoost / GMM-EM.
- **Sub-decision — target shape:** finish-type only (KO/Sub/Dec, corner-agnostic, sidesteps the red confound) vs joint winner×method (6-class, full market benchmark) vs cascade (winner→method). See chat.

### Alternatives still on the table
- **Direction 1 — Winner + odds benchmark** ("predictability ceiling"): binary, simplest, cleanest moneyline benchmark; most saturated topic.
- **Direction 2 — Fighter-style clustering → GMM-EM / DBSCAN:** leakage-free, distinctive, but softer conclusions and no odds angle.

(Full formulation menu in [CLAUDE.md §5](../CLAUDE.md).)

## Open questions blocking the proposal
1. **Target shape** for method-of-victory: finish-type vs winner×method vs cascade.
2. **Extension:** RF vs SAMME-AdaBoost vs GMM-EM.
3. Optional: include **nationality** as an EDA/feature enrichment (needs scraping; gate behind core working)?

## Decision log
- **2026-06-25:** Leaning **method-of-victory** (b); **odds = benchmark**. Dataset → mdabbert. _(Luka)_
- **2026-06-25:** Data inspected — per-method odds exist ⇒ method-of-victory keeps the market benchmark. Baseline module complete (13/13 vs sklearn). _(Claude)_
- **2026-06-25:** Include **who won** (not method-only) — joint winner×method **or** cascade (open; may ask TA). Extension = **SAMME-AdaBoost** *leaning* (learn the multiclass variant for broader value); **RF not ruled out** (soft preference only). Weird outcomes (DQ/CNC/Overturned/NC/Draw + missing-method) → leaning **keep as a 7th "other" class** per Luka (vs drop) — caveat: it's a noisy grab-bag the model won't predict well; the 238 missing-method rows had a real method that's just unrecorded (could recover via join). Nationality (`country` col is event-location, not fighter nationality → needs scrape) + fight-metric regression = optional secondary, gated behind the core. _(Luka)_
- **2026-06-25:** mdabbert leakage audit done — must-exclude `total_fight_time_secs` (current-fight duration) + target cols; rest pre-fight safe. Tests reorganized (units + integration runner, 15/15). _(Claude)_
