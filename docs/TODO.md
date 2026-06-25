# TODO — Luka & Milica

Task list for the team. Legend: **[ ]** open · **[x]** done · **[?]** decision needed (either/or) · **→ depends on** another task.
Owner in **(L / M / both)**. Background/why for most items is in [CLAUDE.md](../CLAUDE.md) and [DATASETS.md](DATASETS.md).

---

## Decisions still open (resolve, or take to the TA)
- [?] **Target shape:** *joint* winner×method (one 6-class model: red/blue × KO/Sub/Dec) **vs** *cascade* (predict winner, then method). Both keep the odds benchmark. → fine to ask the TA. **(both)**
- [?] **Extension:** **SAMME-AdaBoost** (leaning — multiclass AdaBoost, good to learn) **vs** Random Forest (native multiclass, feature importance). **(both)**
- [?] **Weird outcomes** (DQ / No-Contest / Overturned / Draw, + 238 rows with unknown method): keep as a 7th **"other"** class **vs** drop them. Note: the 238 "unknown method" rows had a real KO/Sub/Dec just unrecorded — could be recovered by joining rajeevw/komaksym. **(both)**
- [?] **Logistic regression** as an extra baseline — is it "covered"? It uses the course's gradient descent but isn't its own course topic. Ask TA or skip. **(L or M)**

## Already done (so nobody redoes it)
- [x] Datasets downloaded + inspected; leakage audited → [DATASETS.md](DATASETS.md). Primary = `mdabbert/ufc-master.csv`.
- [x] All course baseline methods implemented from scratch in `src/baselines/` + validated vs sklearn → `tests/` (15/15).
- [x] Repo skeleton, notebook stubs, `requirements.txt`, literature grounding ([SOURCES.md](SOURCES.md)).

---

## Build — in dependency order

### A. Send the proposal (do first — affects max marks)
- [ ] **A1.** Agree the four open decisions above (at least enough to write the proposal). **(both)**
- [ ] **A2.** Send the proposal email to the TA (draft agreed in chat — not stored in repo). → depends on **A1**. **(L)**

### B. Data → features  (`src/data/`)
- [ ] **B1.** Loader for `mdabbert/ufc-master.csv`; drop result columns (`finish*`, `Winner` kept as label, **`total_fight_time_secs`** — it leaks), drop weird outcomes per the **[?]** decision. **(M)**
- [ ] **B2.** Build the feature matrix: keep the pre-built `*_dif` difference features (or build our own red−blue diffs), encode categoricals (stance, weight class), handle debut NaNs. → depends on **B1**. **(M)**
- [ ] **B3.** **Symmetrize corners** (randomly swap A/B so the base rate is 50/50 — kills the "just predict red" shortcut). → depends on **B2**. **(M)**
- [ ] **B4.** **Chronological split** (sort by `date`, train past → test future) + **scale fit on train only**. → depends on **B2**. **(L)**

### C. EDA  (`notebooks/01_eda.ipynb`)  — can start in parallel with B
- [ ] **C1.** Distributions, class balance (method + winner), weight-class / stance breakdowns. **(M)**
- [ ] **C2.** **Red-corner confound analysis** (the differentiator): raw red win-rate → condition on odds/favorite → show colour itself adds little. **(L)**
- [ ] **C3.** Correlations / feature look; sanity-check leakage (debut rows have empty priors). **(L)**

### D. Modelling  (`notebooks/02–03`)  → depends on B
- [ ] **D1.** Run the from-scratch **baseline panel** (Bayes / LDA / QDA / kNN, + a linear floor) on the features. → depends on **B4**. **(both)**
- [ ] **D2.** Implement the **extension from scratch** in `src/extension/` (SAMME-AdaBoost or RF per **[?]**) + a unit test in `tests/unit/`. → depends on **A1**. **(L)**
- [ ] **D3.** **Naive baselines:** always-red and majority-class (always "decision"). **(M)**
- [ ] **D4.** **Odds benchmark:** de-vig the market odds → implied probabilities; compare our model by **log-loss / Brier** (moneyline for winner, per-method props for method). → depends on **D1**. **(L)**

### E. Evaluation & analysis  (`notebooks/04_results`)  → depends on D
- [ ] **E1.** Metrics: accuracy, per-class **F1**, ROC-AUC, **log-loss**, confusion matrix; **mean ± std over seeds**. **(both)**
- [ ] **E2.** **Hyperparameter sweeps** with plots (k for kNN; n_estimators/depth for the extension; PCA dimensions). **(both)**
- [ ] **E3.** **Dim-reduction ablation:** does PCA / LDA-projection help vs full features? + a 2D projection plot coloured by method. **(M)**

### F. Optional (only after the core works)
- [ ] **F1.** Nationality enrichment — scrape per fighter (Sherdog/Wikipedia/Tapology); ablation "does it add signal beyond stats?". Note: mdabbert's `country` is event-location, not nationality. **(either)**
- [ ] **F2.** Fight-metric **regression** (sig. strikes / TD / control time) as a secondary analysis. **(either)**
- [ ] **F3.** Port minor course extras if useful (polynomial regression already added; cost-sensitive Bayes). **(either)**

### G. Deliverables  → depends on E
- [ ] **G1.** Report (intro · problem · dataset · baselines · extension · setup · results · analysis · conclusions). **(both)**
- [ ] **G2.** Oral-defense slides. **(both)**
- [ ] **G3.** Fill in `README.md` for end users. **(either)**
