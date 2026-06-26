# TODO - Luka & Milica

Task list for the team. Legend: **[ ]** open · **[x]** done · **[?]** decision needed (either/or) · **→ depends on** another task.
Owners + priorities are in the Assignments table below. Background/why for most items is in [CLAUDE.md](../CLAUDE.md) and [DATASETS.md](DATASETS.md).

---

## Decisions - LOCKED (TA approved 2026-06-25)
- [x] **Target:** one **joint 6-class** model (winner × method). (TA: cascade has complications the course didn't cover.)
- [x] **Extension:** **SAMME** (TA preference).
- [x] **Baselines:** **LDA, QDA, kNN** (TA: 1-3, with justification → folded into report Section 5.1).
- [x] **Weird outcomes:** **dropped** (rare, not meaningfully predictable; explained in the report).
- [x] **Split:** chronological ~80/20. **Logistic regression:** not used (enough baselines).
- ⚠️ **Deadline = tomorrow; no late commits accepted.** Out of scope (TA: nothing beyond the proposal): nationality scrape, fight-metric regression.

## Task sequence (dependency graph)

Issues are [#1-#18](https://github.com/lcubrilo/PRML-Projekat/issues). Arrows = "must finish before". GitHub renders the diagram below.

Node colour = priority: red P0 critical, yellow P1 important, green P2 nice-to-have (matches the GitHub labels). Green check = done.

```mermaid
graph LR
  classDef p0 fill:#f4a8a8,stroke:#b60205,color:#000;
  classDef p1 fill:#fde79a,stroke:#9c8400,color:#000;
  classDef p2 fill:#bfe6c4,stroke:#0e8a16,color:#000;
  classDef done fill:#cfeccf,stroke:#393,color:#000;
  A2["A2 proposal ✅"]:::done
  B1["B1 load + 6-class label"]:::p0 --> B2["B2 features"]:::p0
  B2 --> B3["B3 symmetrize corners"]:::p0
  B3 --> B4["B4 chrono split + scale"]:::p0
  B2 --> C1["C1 EDA: distributions"]:::p0
  B2 --> C2["C2 EDA: red-corner confound"]:::p1
  B2 --> C3["C3 EDA: leakage check"]:::p2
  B4 --> D1["D1 baselines LDA/QDA/kNN"]:::p0
  B4 --> D2["D2 SAMME from scratch"]:::p0
  B1 --> D3["D3 naive baselines"]:::p1
  D1 --> D4["D4 odds benchmark"]:::p1
  D1 --> E1["E1 metrics"]:::p0
  D2 --> E1
  D1 --> E2["E2 hyperparam sweeps"]:::p1
  D2 --> E2
  D1 --> E3["E3 DR ablation"]:::p2
  E1 --> G1["G1 report"]:::p0
  D4 --> G1
  E2 --> G1
  E3 --> G1
  G1 --> G2["G2 slides (last)"]:::p1
```

Critical path (all P0): B1 -> B2 -> B3 -> B4 -> D1/D2 -> E1 -> G1.
EDA (C1-C3), D3 (naive), and D4 (odds) hang off earlier nodes and can run in parallel.

### Execution segments (who waits on whom)

The pipeline **B1→B2→B4 is Milica's and B2 gates almost all of Luka's work**. The only big task with no data dependency is **D2 (SAMME)** - Luka builds it against iris/synthetic data while Milica clears the pipeline. Late on, **E1/E2 need both D1 and D2 (both Luka's)**, so the wait flips: Milica waits on Luka there.

| Segment | Milica | Luka | 🔁 Sync gate |
|---|---|---|---|
| **1** parallel | B1 → B2 | D2 SAMME (synthetic data) | **B2 done** → unblocks all EDA + B3/B4 |
| **2** after B2 | B3 → B4; D3 anytime | C1/C2/C3 EDA; finish D2 | **B4 done** → unblocks D1 |
| **3** after B4 | E3 ablation; start report-factual | D1 baselines → D4 odds | **D1 *and* D2 done** → unblocks E1/E2 |
| **4** eval | E2 sweeps, E3 | E1 metrics | all eval done → report |
| **5** deliverables | report-factual, G3 README | report-narrative | report done → G2 slides (both) |

**Tightest squeeze = Segment 3:** E1/E2 need both of Luka's D1+D2. Luka should have **D2 functionally done by the B4 gate** so D1 can run the moment B4 lands and Milica isn't left idle.

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
| [#14](https://github.com/lcubrilo/PRML-Projekat/issues/14) | E2 hyperparam sweeps | P1 | Milica (tentative) |
| [#11](https://github.com/lcubrilo/PRML-Projekat/issues/11) | D3 naive baselines | P1 | Milica |
| [#17](https://github.com/lcubrilo/PRML-Projekat/issues/17) | G2 slides (last) | P1 | both (tentative) |
| [#8](https://github.com/lcubrilo/PRML-Projekat/issues/8) | C3 leakage check | P2 | Luka |
| [#15](https://github.com/lcubrilo/PRML-Projekat/issues/15) | E3 DR ablation | P2 | Milica (tentative) |
| [#18](https://github.com/lcubrilo/PRML-Projekat/issues/18) | G3 README | P2 | Milica (tentative) |

**Domain-knowledge notes:** Luka knows UFC, Milica is new to it, so interpretation goes to Luka. B2 (#3): Luka shortlists which features matter, Milica implements. Report (#16) is split by domain: Milica writes the factual sections (dataset, setup, baselines), Luka writes the narrative/analysis (intro, problem, EDA, analysis, conclusions) - see `report/report.md`.

## Already done - foundations (not part of the B-G plan below; live status lives in the checkboxes)
- [x] Datasets downloaded + inspected; leakage audited → [DATASETS.md](DATASETS.md). Primary = `mdabbert/ufc-master.csv`.
- [x] All course baseline methods implemented from scratch in `src/baselines/` + validated vs sklearn.
- [x] Repo skeleton, notebook stubs, `requirements.txt`, literature grounding ([SOURCES.md](SOURCES.md)).
- Reusable from-scratch infra built this round is tracked at its own B-G task below (SAMME, metrics, naive baselines, odds benchmark, plotting). Full test suite **44/44** via `tests/validate_baselines.py`.

---

## Build - in dependency order

### A. Send the proposal (do first - affects max marks)
- [x] **A1.** Agree the four open decisions above (at least enough to write the proposal).
- [x] **A2.** Send the proposal email to the TA (draft agreed in chat - not stored in repo). → depends on **A1**.

### B. Data → features  (`src/data/`) - Milica
- [x] **B1.** Loader + 6-class label (`load.py`): drops result/leak cols and weird outcomes. Done.
- [x] **B2.** Feature matrix (`features.py::make_features`): `*_dif` + absolute R_/B_ cols, one-hot stance/weight_class, debut NaNs filled. Market cols (`MARKET_COLS` + any `odds`) now excluded (benchmark-only). 112 features.
- [x] **B3.** Symmetrize corners (`features.py::symmetrize`): negate diffs, swap R_/B_, flip label side. Done.
- [x] **B4.** Chronological split + scale-fit-on-train-only (`split.py`). Done (Milica; fixed a duplicate-def bug that broke import). Verified end-to-end: 5529/1382 chronological, train scaled to mean 0/std 1, one-hots left untouched. **Unblocks all modelling.** (Still pending: move make_features median imputation to train-only - separate from B4.)

### C. EDA  (`notebooks/01_eda.ipynb`) - Luka
- [x] **C1.** Class balance (6-class + winner + method), method-by-weight-class, feature distributions. Figures saved. Done.
- [x] **C2.** **Red-corner confound analysis** (the differentiator): raw red 57.7% → conditioned on the favourite it is ~70% (red fav) vs ~40% (blue fav); favourite wins 65.5% regardless of corner. Done.
- [x] **C3.** Correlation heatmap of key diff features (reach~height and ko~win at 0.63; otherwise low). Leakage sanity check passed: experienced fighters 0% missing priors vs debutants ~75% missing -> aggregates are pre-fight. Done.

### D. Modelling  (`notebooks/02-03`)
- [x] **D1.** Baseline panel on the split (`02_baselines.ipynb`). LDA best: 6-class 0.355, winner-collapse **0.630** (> always-red 0.562). QDA(reg=1.0)/kNN weaker (high-dim). (Luka)
- [x] **D2.** SAMME from scratch + tests. Run on the real split in `03_extension.ipynb` (~9 min fit).
- [x] **D3.** Naive baselines (`02_baselines.ipynb`): majority 0.255, always-red (winner, orig corners) 0.562, coin-flip 0.500. (Luka)
- [~] **D4.** Odds benchmark (`03_extension.ipynb`): SAMME proba vs de-vigged market, log-loss/Brier. Running with the SAMME fit. (Luka)
- Pipeline composed in `src/data/pipeline.py::build_dataset` (leakage-safe, incl. train-only imputation fix). QDA/LDA/kNN now expose `predict_proba`.

### E. Evaluation & analysis  (`notebooks/04_results`)  → depends on D
- [ ] **E1.** Metrics - **functions done** (`src/metrics.py`: accuracy, F1, ROC-AUC, log-loss, Brier, confusion matrix, seed summary); apply to the baseline + SAMME outputs. Needs D1/D2 runs. (Luka)
- [ ] **E2.** Hyperparameter sweeps + plots - plotting helpers + `staged_score` ready; needs models on data. (Milica)
- [ ] **E3.** Dim-reduction ablation (PCA / LDA-projection) + 2D plot coloured by method. (Milica)

### F. Out of scope (TA: nothing beyond the proposal; deadline tomorrow)
- ~~F1 nationality scrape~~ · ~~F2 fight-metric regression~~ - cut. Don't start these.

### G. Deliverables  → depends on E
- [ ] **G1.** Report - **drafted: Sections 1, 1.1, 2, 5.1, 5.2, 9-future-work, References.** Remaining: 3 dataset, 4 EDA, 6 setup, 7 results, 8 analysis, 9 findings. (both)
- [ ] **G2.** Oral-defense slides. (both)
- [ ] **G3.** README for end users. (Milica)

## Backlog (later / nice-to-have, not blocking)
- [ ] **SAMME speed.** ~2.8s/stump on the full data (5529x112) -> ~9.4 min for 200 estimators. Cause: the from-scratch stump scans every candidate threshold (~n_samples) for all 112 features each round. Fine for one fit + `staged_score`; if seed-averaging/experimentation gets painful, speed it up via a quantile threshold grid (~32-64 thresholds/feature) - keep Luka's comments where they still apply.
- [ ] **Cost-sensitive option.** Decision-time Bayes-risk with a winner/method cost matrix (partial credit for right-winner-wrong-method) - works for all models via predict_proba; report standard + cost-sensitive separately. Stretch: bake the cost into SAMME's training (AdaCost-style) and see if it helps.
- [ ] **Regularization/PCA finding.** Unregularized QDA fails in 112-dim (worse than majority, log-loss 11); reg helps log-loss but QDA still < LDA -> motivates PCA (E3). Good report finding: necessity of regularization/DR in this regime.
