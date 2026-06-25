# TODO — Luka & Milica

Task list for the team. Legend: **[ ]** open · **[x]** done · **[?]** decision needed (either/or) · **→ depends on** another task.
Owner in **(L / M / both)**. Background/why for most items is in [CLAUDE.md](../CLAUDE.md) and [DATASETS.md](DATASETS.md).

---

## Decisions — LOCKED (TA approved 2026-06-25)
- [x] **Target:** one **joint 6-class** model (winner × method). (TA: cascade has complications the course didn't cover.)
- [x] **Extension:** **SAMME** (TA preference).
- [x] **Baselines:** **LDA, QDA, kNN** (TA: 1-3, with justification → `report/baselines_chosen.md`).
- [x] **Weird outcomes:** **dropped** (rare, not meaningfully predictable; explained in the report).
- [x] **Split:** chronological ~80/20. **Logistic regression:** not used (enough baselines).
- ⚠️ **Deadline = tomorrow; no late commits accepted.** Out of scope (TA: nothing beyond the proposal): nationality scrape, fight-metric regression.

## Assignments & priorities (live mirror = GitHub issues)
P0 critical · P1 important · P2 nice-to-have. Early tasks are firm; later ones (E/G) are tentative and can be reshuffled.

| Issue | Task | Prio | Owner |
|---|---|---|---|
| [#2](https://github.com/lcubrilo/PRML-Projekat/issues/2) | B1 load + 6-class label | P0 | **Milica** |
| [#3](https://github.com/lcubrilo/PRML-Projekat/issues/3) | B2 features | P0 | **Milica** |
| [#4](https://github.com/lcubrilo/PRML-Projekat/issues/4) | B3 symmetrize | P0 | **Milica** |
| [#5](https://github.com/lcubrilo/PRML-Projekat/issues/5) | B4 split + scale | P0 | **Milica** |
| [#10](https://github.com/lcubrilo/PRML-Projekat/issues/10) | D2 SAMME from scratch | P0 | **Luka** |
| [#6](https://github.com/lcubrilo/PRML-Projekat/issues/6) | C1 EDA distributions | P0 | **Luka** |
| [#13](https://github.com/lcubrilo/PRML-Projekat/issues/13) | E1 metrics | P0 | **Luka** |
| [#9](https://github.com/lcubrilo/PRML-Projekat/issues/9) | D1 run baselines | P0 | **Luka** |
| [#16](https://github.com/lcubrilo/PRML-Projekat/issues/16) | G1 report | P0 | both |
| [#7](https://github.com/lcubrilo/PRML-Projekat/issues/7) | C2 red-corner confound | P1 | Luka |
| [#12](https://github.com/lcubrilo/PRML-Projekat/issues/12) | D4 odds benchmark | P1 | Luka |
| [#14](https://github.com/lcubrilo/PRML-Projekat/issues/14) | E2 hyperparam sweeps | P1 | Luka (tentative) |
| [#11](https://github.com/lcubrilo/PRML-Projekat/issues/11) | D3 naive baselines | P1 | Milica |
| [#17](https://github.com/lcubrilo/PRML-Projekat/issues/17) | G2 slides (last) | P1 | both (tentative) |
| [#8](https://github.com/lcubrilo/PRML-Projekat/issues/8) | C3 leakage check | P2 | Luka |
| [#15](https://github.com/lcubrilo/PRML-Projekat/issues/15) | E3 DR ablation | P2 | Milica (tentative) |
| [#18](https://github.com/lcubrilo/PRML-Projekat/issues/18) | G3 README | P2 | Milica (tentative) |

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

### F. Out of scope (TA: nothing beyond the proposal; deadline tomorrow)
- ~~F1 nationality scrape~~ · ~~F2 fight-metric regression~~ — cut. Don't start these.

### G. Deliverables  → depends on E
- [ ] **G1.** Report (intro · problem · dataset · baselines · extension · setup · results · analysis · conclusions). **(both)**
- [ ] **G2.** Oral-defense slides. **(both)**
- [ ] **G3.** Fill in `README.md` for end users. **(either)**
