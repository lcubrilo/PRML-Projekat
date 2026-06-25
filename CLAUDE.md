# PRML Project - Knowledge Base

Course project for **Pattern Recognition and Machine Learning / Препознавање облика и машинско учење** (PMF UNS). Team: **Luka** (lcubrilo01) & **Milica**.

This is the **knowledge base** - stable facts, constraints, domain knowledge, and the full option space. It does not track decisions or tasks:
- Live decisions + proposal status → **[PROPOSAL.md](docs/PROPOSAL.md)**
- Team tasks → **[TODO.md](docs/TODO.md)**

Ground truth for the assignment is `Project context/Project requirements.pdf` (the TA's brief). Claims below tagged **⚠️ VERIFY** are from prior research/chats and have not been independently confirmed - fact-check before relying on them in the report or proposal.

---

## 1. The assignment (fact-checked against the PDF)

A team designs and executes a project around a real-world ML problem, with two mandatory parts:

**Comparative study** - (a) select real-world dataset(s), (b) clearly define the ML problem, (c) perform EDA, (d) apply and compare method(s) covered in the course, (e) **implement the chosen baseline methods from scratch**, (f) evaluate and interpret performance.

**Extension component** - additionally implement **from scratch** one of:
- a classical ML algorithm **not** covered in the course, OR
- an existing **extension/improvement** of a covered algorithm.

> The method must already exist in literature/established practice and be implemented from scratch (like in exercises). **External libraries may only be used for validation/comparison, not as the primary implementation.**

**Critical reading: the from-scratch rule covers the baselines too, not just the extension** (point (e) is explicit). Everything you *report* must come from your own code; sklearn et al. only prove correctness. The intended pattern: show your implementation matches a library's output, then produce all reported numbers from your own code.

The extension needs a **citable reference** (paper/article/textbook) - you reimplement a known method, you don't invent one.

**TA's suggested extension algorithms** (verbatim): SVM, Random Forests, Gradient Boosting, AdaBoost, DBSCAN, Weighted kNN, Kernel PCA, Adaptive KDE, Regularized LDA, k-means++. (Note the split: the first five are *new* algorithms; the last five are *upgrades* of covered methods.)

**TA's five suggested approach framings** (pick one as the spine of the comparative study):
1. one dataset, many algorithms;
2. one algorithm, many datasets;
3. study one advanced extension in depth;
4. compare baseline methods and modern extensions;
5. investigate algorithmic behavior under different data regimes.

**TA's example datasets:** Wine, Breast Cancer Wisconsin, Dry Bean, Sonar, Titanic, Mall Customers, MNIST subset. Datasets are the team's free choice but **subject to instructor approval** (UFC is not on this list → approval matters).

### Proposal (advisable, treat as required)
Short proposal to the TA with six points: (a) dataset(s), (b) problem formulation, (c) planned baseline algorithms, (d) chosen independent extension, (e) reference source, (f) preliminary implementation plan. The brief warns: **if the proposal isn't sent for approval, full marks aren't guaranteed.**

### Deliverables
- Full report: introduction, problem description, dataset description, baseline, extension method(s), experimental setup, results, analysis, conclusions.
- Jupyter notebook(s) and/or Python script(s) with all implementations - **via GitHub** (this repo).
- Presentation for oral defense.

### The PDF's worked example (one valid template, not a mandate)
*"Comparative Study of Classical and Extended Classification Methods on the Breast Cancer Wisconsin Dataset"* - baselines QDA + kNN; extension weighted kNN / SVM from scratch; RQs: does the extension beat baselines? does PCA help? complexity vs. gain? **Listed expected outputs:** EDA, method implementations, performance comparison, **hyperparameter analysis**, visualizations, discussion. (These outputs are from the example, so treat hyperparameter analysis + visualizations as strongly expected, though only EDA is named a hard requirement in (c).)

---

## 2. Course methods already implemented (reuse freely)

The from-scratch coursework lives in `Project context/course_materials/`. The directory names alone enumerate every covered topic (each has a solved `.ipynb`):

`02_linear_regression` (+ gradient descent) · `03_least_squares` (perceptron / Ho–Kashyap) · `04_bayes_decision` (Bayes classifier) · `05_quadratic_classifiers` (QDA) · `06_mle` · `07_kde` · `08_knn` · `09_pca` · `10_lda` · `11_clustering` (k-means). (`01_intro` = linear algebra + probability refreshers.)

**Reuse note:** the base techniques are now **ported into `src/baselines/`** (importable) and validated against sklearn (`tests/`). Reused code must still be **re-validated on the actual project dataset** - don't assume correctness carries over.

**Strategic consequence of free baselines:** the baseline tier costs ~nothing, so run a *wide* baseline panel (the brief rewards a richer comparative study). All real effort/risk concentrates in the **extension** - the one thing still genuinely from scratch.

---

## 3. Chosen domain: UFC / MMA fight analysis

### The "predictability ceiling" framing (the project's strongest narrative)
**Verified (see [SOURCES.md](docs/SOURCES.md)):** published UFC winner-prediction **clusters around ~63–67% accuracy** - a single clean strike ends a fight, so the sport is high-variance. It's a *typical ceiling, not an absolute wall* (careful ensembles reach high-60s; one SMOTE+time-split model claims ~79% - treat as leakage-suspect). For a course graded on *rigor, not on beating the world*, this is an asset: frame as *how close can we get to the typical ceiling*, against reference points:
- coin flip → 50%
- always-pick-red → ~58–64% (inflated by a confound, below; komaksym shows 64% red)
- betting market → ~65% (the practical ceiling on extractable signal, for *winner* prediction)

A final ~64% then reads as a *finding* ("we've about hit the limit; the rest is irreducible randomness"), and a low number can't cost points.

### Two methodology traps (what a grader checks)
1. **Lookahead leakage.** Career-aggregate stats are only safe if computed *up to but excluding* the bout being predicted. Weak public projects quietly train on in-fight stats and report 85–95% - that's leakage. Verify pre-fight integrity: a debut row should have null/zero priors; aggregates should accumulate fight-to-fight; a fighter entering 5–0 must not already show 6 wins.
2. **Red-corner confound.** Red wins ~58% raw, but "red" is systematically the favorite/champion - a selection effect, not a color effect (advantage has waned toward 50% over time **⚠️ VERIFY**). Demonstrating this as a *controlled* EDA analysis (raw rate → condition on favorite/champion → show color itself does little) is sophisticated and rare. Modeling fix: **symmetrize** corners (randomize A/B to force a 50/50 base rate) and use **difference features** (stat_A − stat_B).

These two are the project's differentiators - most public UFC projects (hundreds, nearly all sklearn/XGBoost winner-prediction) ignore them.

### Domain background & published findings (EDA fodder + related-work for the report)
All **⚠️ VERIFY** (sourced from research chat, mix of academic and journalistic):
- **FightMetric / UFC Stats** (Rami Genauer, ~2007; official since 2008) - the hand-coded ~67-category dataset everything descends from. Flagship demo: GSP's takedown accuracy highest in history.
- **Fightnomics** (Reed Kuhn, 2013) - "Moneyball of MMA." Claims: age matters only past a ~4-year gap; reach matters past ~2 inches.
- **Southpaw paradox** - left-handers overrepresented in interactive combat sports and fight more often, but win-rate difference vs orthodox is *not* statistically significant (PLOS One 2013: 64.0% vs 62.6%).
- **Weight-cutting** - genuinely contested across studies; latest reconciliation: regain % doesn't predict *who* wins but may shape *how* (more regain → more KO/TKO; less → more decisions).
- **Reach** - disputed: Fightnomics says it matters; several ML/EDA projects found weak/no correlation with outcomes or strikes landed.
- **Betting-market efficiency** - repeatedly the strongest single predictor; ML struggles to beat it (the reason it's used as a benchmark, §6).
- **Wrestling/grappling base & the Dagestan pipeline** - reaffirmed statistically (control time, takedown accuracy) and journalistically (mostly qualitative - *not* formal DS).
- **JudgeAI** (LiteralFightNerd) - predicts judges' round scores from scorecards; red won 57.6% of rounds, 10-8 rounds only 3.3%.
- **Elo / Meta UFC Rankings** - Elo argued unsuitable (fighters compete too infrequently to stabilize); UFC's official **Meta UFC Rankings** (Elo-based, ~June 2026) replaced media voting.
- Viral *qualitative* analysis (film study, not DS): **Lawrence Kenshin**, **Jack Slack**, **BJJ Scout**.

---

## 4. Datasets

Effectively **one canonical raw source** - `ufcstats.com` (successor to FightMetric) - repackaged endlessly. Don't combine sources yourself; the useful merges already exist. **Specifics below are now verified by inspection → see [DATASETS.md](docs/DATASETS.md). Chosen primary: `mdabbert` (one file: target + pre-fight features + odds).**

| Source | What / when | Odds? | Use |
|---|---|---|---|
| **`mdabbert/ultimate-ufc-dataset`** (Kaggle) | Superset: Warrier stats + odds (moneyline **+ per-method props**) + rankings; 2010–2026, 118 cols, `finish` target. | **Yes** | ✅ **CHOSEN PRIMARY** - one file does it all. |
| **`rajeevw/ufcdata`** (Kaggle) | Canonical base, ~1994–2021. `data.csv` ships pre-fight aggregates + `Winner`; method (`win_by`) in `raw_total_fight_data.csv`. Leakage-safe by design. | No | Fallback (no odds); larger history. |
| **`WarrierRajeev/UFC-Predictions`** (GitHub) | The *generator* behind rajeevw: scraper + authoritative column defs + a reference RF/XGBoost baseline (~0.72 acc, inflated by the red-corner imbalance). | - | Reference, not a separate dataset. |
| **`komaksym/UFC-DataLab`** (GitHub) | Most modern/best-engineered; OCR-parses judges' scorecards (PaddleOCR); updated quarterly. | - | Only for a **judging/scorecard** project. |
| `neelagiriaditya`, `cadelueker`, `rajaisrarkiani`, `fatismajli` (Kaggle) | Same ufcstats lineage, through 2025. **Audited 2026-06-25 → confirmed REDUNDANT** (see [DATASETS.md](docs/DATASETS.md)): no odds, no nationality, in-fight stats only, and the fighter career-rate table they ship already exists in our `raw_fighter_details.csv`. mdabbert is fresher (2026-03). | **No** | Skip - nothing they add. |
| **Tapology** (web, no clean CSV) | Fighter bios - nationality, gym/team, broader history - richer on *who* the fighter is than ufcstats. | No | Only if you want nationality/team/martial-art-base features; **requires scraping**, out of scope unless pursued. |

**Columns - now verified by inspection (see [DATASETS.md](docs/DATASETS.md)):** the lineage carries fighter **names**, **weight class**, **date/location**, **method** of victory, **round**+time, per-fighter **fight stats** (strikes by target/position, TD, sub att, control time, KD). **Nationality / martial-art base / gym / coach are NOT present anywhere** → need Tapology/Sherdog scraping. Warrier *preprocessed* anonymizes to `R_`/`B_` columns + ships pre-fight aggregates; komaksym `stats_raw` has **in-fight** stats (must self-aggregate to be leakage-safe); komaksym `stats_processed` uses **winner−loser deltas** (leaks the winner).

**Oversampling - CORRECTED:** not baked into the dataset; it's a modeling-notebook step (`sklearn resample` in Warrier's `xgboost-Oversampled.ipynb`). The base CSV is ~one-row-per-fight, so the "dedupe before split" caution only applies to an oversampled variant. (Details + the method-class distribution and ~64% red skew in [DATASETS.md](docs/DATASETS.md).)

For **fighter-style clustering** you need a *fighter-level* table (one row per fighter, career-aggregated) - built by aggregating any fight-level file.

---

## 5. Option space (pick one cell per column → see PROPOSAL.md)

**Problem formulation binds everything** - it determines which baselines and extensions are legal.

### Problem formulations (a–z of what the chats raised)
| Formulation | Task | Notes |
|---|---|---|
| **Winner** (red vs blue) | binary clf | Standard/expected; most saturated publicly → only worth it with the leakage-audit + odds-benchmark twist. |
| **Method of victory** (KO/TKO / sub / decision) | multiclass clf | Rarely done publicly, more learnable from style features, meaningful confusion matrix. **Strong.** |
| **Finish vs decision** ("goes the distance") | binary clf | Often more learnable than winner. |
| **Round of finish** | multiclass clf | Data-starved fast → stretch goal, not co-equal target. |
| **Cascade: who + how + which round** | stacked | Appealing but really 3 models; round model is the weak link. Scope risk. |
| **Fighter-style clustering** | clustering | **Leakage-free** (no labels/corners), mirrors the ACM 2024 style-clustering paper, interpretable. **Strong + distinctive.** |
| **Fight/round-type clustering** (grappling- vs striking-heavy bouts) | clustering | Less explored; softer conclusions. |
| **Regression** (fight duration / sig. strikes / control time) | regression | Only framing that exercises the linear-regression/GD baseline. Underused but valid. |
| **Judging / scorecard prediction** (JudgeAI-style) | clf/regression | Least-saturated, most distinctive; **needs `komaksym` scorecard data** + messier pipeline. |
| **Beat-the-market** | layer on winner | Benchmark model vs de-vigged odds (§6). |
| **Anomaly / upset detection** | unsupervised | Thin as a *primary* narrative (no clean ground-truth label for "upset"); better as a sub-analysis. |
| **DR-centric** (PCA/LDA as the star, Kernel PCA extension) | - | Better as an ablation inside another project than as the whole story; Kernel PCA only shines if linear PCA visibly fails. |

**Avoid: fighter career-trajectory with external context** (injuries, camps, coaches, teams). That data isn't in ufcstats; bespoke scraping is noisy, hard to validate, and earns no methodology points. Great real project, bad deadline project.

### Baselines (course-covered, from scratch)
- **Classification:** Gaussian/Naive Bayes + LDA + QDA + kNN (clean linear-vs-quadratic-vs-instance-based contrast). Add perceptron / least-squares for a linear floor. KDE/Parzen-window classifier if density-based.
- **Clustering:** k-means. **Regression:** least-squares + gradient descent (compare the two).
- **⚠️ VERIFY:** chats also suggested *logistic regression via gradient descent* as a baseline - but logistic regression is **not** an explicit course directory, so confirm whether it counts as "covered" before using it as a baseline (it leans on the covered GD machinery, but may not qualify).

### Extensions (from scratch, must match the task) - cost depends on whether it upgrades a base you already have
- **Cheap (reuse homework code, ~a day):** weighted kNN (←kNN), regularized/shrinkage LDA (←LDA), k-means++ (←k-means), Kernel PCA (←PCA). Risk: can look *thin* given baselines were free → compensate with experimental depth.
- **Medium (write fresh - the sweet spot):** **AdaBoost** w/ decision stumps, **DBSCAN**, **GMM-via-EM**. Clean to code, well-referenced, meaningfully different from baselines.
- **Hard (from zero):** kernel **SVM** (needs SMO; use **Pegasos** sub-gradient to stay sane), Random Forest (build CART first, then bag), Gradient Boosting, spectral clustering.
- **Other candidates raised in chats:** mean-shift clustering (built on KDE - code reuse), Mahalanobis/metric kNN (a meatier kNN upgrade than distance-weighting), Adaptive KDE, ridge/LASSO (regularized linear regression) or SVR (regression route).

> **Meaningfulness constraint:** a kernel method / DBSCAN only earns its place on data the baseline genuinely can't handle. UFC outcome data is **roughly linear in difference-features (⚠️ VERIFY)**, so kernel tricks may show *little gain* - fine if the RQ is explicitly "does the added complexity even pay off?", but don't expect fireworks. Having kNN done does **not** help you write an SVM.

### References (pair to the chosen extension)
SVM: Cortes & Vapnik (1995); Pegasos: Shalev-Shwartz et al. (2011). AdaBoost: Freund & Schapire (1997). Random Forest: Breiman (2001). Gradient Boosting: Friedman (2001). DBSCAN: Ester et al. (1996). k-means++: Arthur & Vassilvitskii (2007). GMM/EM: Dempster, Laird & Rubin (1977); Bishop PRML ch.9. Weighted kNN: Dudani (1976). Kernel PCA: Schölkopf, Smola & Müller (1998). Regularized LDA: Friedman (1989) / Ledoit & Wolf (2004). Adaptive KDE: Silverman (1986) / Abramson (1982). Mean-shift: Comaniciu & Meer (2002) **⚠️ VERIFY**. Spectral clustering: Ng, Jordan & Weiss (2002) **⚠️ VERIFY**.
Domain framing (motivation, not the from-scratch source): Kuhn's *Fightnomics*; the **ACM 2024** factor-analysis + K-means paper *"Artificial Intelligence in UFC Outcome Prediction..."* (`10.1145/3696952.3696966`). **Citation discrepancy RESOLVED:** the "IEEE ensemble"/"IEEE style-clustering" notes were one **mislabeled ACM paper** (see [SOURCES.md](docs/SOURCES.md)).
General texts: Duda/Hart/Stork (*Pattern Classification*), Bishop (*PRML*), Hastie/Tibshirani/Friedman (*ESL*).

---

## 6. Betting odds - two distinct uses (keep separate)

- **As a benchmark (recommended):** the actual fight outcome is *always* the ground truth you score against; odds are a *rival forecaster*, not a substitute for truth. The market aggregates sharp money + bookmaker models + insider info, so it's the practical ceiling on extractable signal; landing within a point or two is the strong conclusion. Convert American odds → implied probability and **de-vig** (the two implied probs sum to >100% due to the overround; normalize). Because the market outputs a probability, compare with **log-loss / Brier score**, not just accuracy.
- **As input features (careful):** closing odds encode nearly everything → a model with odds-as-features just relearns the market and flattens the story. Keep odds in the benchmark, out of the feature set. (Closing odds are also slightly future-aware; opening odds are cleaner if distinguished.)
- **Per-method prop odds exist (verified).** `mdabbert/ufc-master.csv` ships not just moneyline (`R_odds`/`B_odds`) but **per-method props** (`r_ko_odds`/`r_sub_odds`/`r_dec_odds` + blue, ~82% coverage). ⇒ method-of-victory / finish-type prediction **can** be benchmarked against the market too, not only winner prediction. (Earlier "odds are moneyline-only" note was wrong.)
- **Kalshi / Polymarket** run live UFC markets but aren't a clean historical archive - use `mdabbert`'s built-in sportsbook odds for the benchmark; live markets are at most a small demo garnish.

---

## 7. Implementation plan skeleton

1. Acquire + clean (missing stats, debut fighters). **Dedupe** the oversampling.
2. **EDA (mandatory)** - distributions, the red-corner confound analysis, correlations, class balance, weight-class/stance breakdowns.
3. **Leakage-safe features** - symmetrize corners + difference features; verify pre-fight integrity (debut-row / accumulation checks). Safest fallback: static-only fields (age, height, reach, stance, windowed record).
4. Preprocess + split - **scale fit on train only** (critical for kNN/SVM/k-means/PCA). Prefer a **chronological** split (train past → test future) over random for honesty.
5. Implement baselines (reuse module), then the extension, all from scratch.
6. **Validate against sklearn** (correctness only; report own numbers).
7. **Hyperparameter analysis (expected output)** - sweep k / C&γ / n_estimators&depth / eps&minPts / K, with plots.
8. Evaluate - clf: accuracy, F1, ROC-AUC, **log-loss**, confusion matrix; clustering: silhouette, inertia, ARI vs stance/weight-class pseudo-labels; regression: RMSE/MAE/R². Report **mean ± std over multiple seeds**.
9. Optional odds benchmark (log-loss vs de-vigged market).
10. Analysis + report + slides.

---

## 8. Status of earlier open tasks (see [TODO.md](docs/TODO.md) for live tasks)
- ✅ Datasets downloaded + inspected → [DATASETS.md](docs/DATASETS.md).
- ✅ Baselines ported into `src/baselines/` + validated vs sklearn (`tests/`).
- ✅ Citation resolved (ACM, not IEEE) + ~63–67% ceiling grounded → [SOURCES.md](docs/SOURCES.md).
- ⬜ Still open: does **logistic regression** count as a "covered" baseline? (not an explicit course dir.)

---

## 9. Deliverable shape & tentative repo structure

**No UI is required.** Deliverables are: report + notebooks/scripts (GitHub) + oral-defense presentation. Recommended shape - **from-scratch implementations in an importable `src/` module, narrative + results in notebooks that import from it** (satisfies "notebook(s) and/or scripts"; keeps the from-scratch code reusable, testable, and validated once against sklearn). An optional lightweight **demo cell/function** ("pick two fighters → predicted outcome") is a nice defense touch but not required; treat its year/weight-class what-ifs as *correlational*, not causal (the model doesn't know the true effect of changing weight class - caveat it).

Current layout:
```
src/baselines/      from-scratch course methods ✅ (validated vs sklearn)
src/extension/      the from-scratch extension (TODO: SAMME-AdaBoost)
src/data/           loading, cleaning, leakage-safe feature build (TODO)
notebooks/          01_eda  02_baselines  03_extension  04_results (stubs)
tests/              unit/ + validate_baselines.py (integration runner)
data/raw, data/processed   (gitignored)
report/  slides/    write-up + presentation
docs/               PROPOSAL · TODO · DATASETS · SOURCES
```

---

## Appendix - pre-UFC general candidates (in case of pivot)
Earlier brainstorming (before converging on UFC) surfaced non-UFC options worth keeping on record:
- **Datasets:** the TA's list (Wine, Breast Cancer, Dry Bean, Sonar, Titanic, Mall Customers, MNIST subset) + Iris, sklearn Digits (8×8, "MNIST subset" alt), Ionosphere (binary, non-linear), synthetic two-moons / concentric circles (controllable non-linear regime).
- **Generic blueprints:** regularization story (Sonar/Breast Cancer + LDA/QDA → regularized LDA); clustering story (Mall Customers + k-means → k-means++ / DBSCAN); kernel story (Sonar + linear/kNN → RBF-SVM); ensembles story (Dry Bean/Sonar + QDA/kNN → AdaBoost / Random Forest).
