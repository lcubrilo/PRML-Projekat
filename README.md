# UFC Fight Outcome Prediction

A Pattern Recognition and Machine Learning course project (PMF UNS). We predict the joint outcome of UFC fights, which corner wins and how (KO/TKO, submission, or decision), from pre-fight information, using classifiers implemented from scratch and benchmarked against the betting market.

Team: Luka Cubrilo, Milica Cvetic.

## What this project does
- Predicts a six-class target (winner x method) for roughly 6,900 UFC fights (2010-2026) from leakage-safe pre-fight features.
- Implements the baselines (LDA, QDA, kNN) and the extension (SAMME multiclass AdaBoost) from scratch in NumPy; scikit-learn is used only to validate correctness, never to produce a reported number.
- Audits for lookahead leakage, controls the red-corner confound, and benchmarks against the de-vigged betting market by log-loss.

## Key findings
- The best models reach about 63% winner-prediction accuracy, inside the published 63-67% ceiling for this problem.
- The boosting extension (SAMME) ties the simple linear discriminant (LDA): the pre-fight signal is close to linear, so the added complexity does not pay off, and dimensionality reduction does not rescue the weaker methods either.
- The betting market is still a slightly better forecaster (log-loss 1.55 versus our 1.66), but we come close.
- The "red corner advantage" is a selection effect (the red corner is usually the favourite), not a real edge.
- Recent form, grappling, and experience carry the most signal; reach and age carry little. Predictability varies by division (welterweight ~0.73 down to flyweight ~0.55), but not in any clean relationship with weight.

## Repository layout
```
src/baselines/   from-scratch course methods (LDA/QDA, kNN, PCA, ...) + naive baselines
src/extension/   SAMME multiclass AdaBoost
src/data/        loading, feature build, chronological split, leakage-safe pipeline, odds
src/metrics.py   from-scratch evaluation metrics
src/plotting.py  figure helpers
notebooks/       01_eda  02_baselines  03_extension  04_results
tests/           unit tests + integration runner (validated against scikit-learn)
report/          write-up and figures
data/raw/        the dataset (mdabbert Ultimate UFC Dataset)
```

## Running it
1. Install dependencies: `pip install -r requirements.txt`
2. Run the test suite: `python tests/validate_baselines.py`
3. Open the notebooks in order (01 to 04); each imports from `src/` and reproduces its results and figures.

## Data
mdabbert Ultimate UFC Dataset (Kaggle), a pinned snapshot committed under `data/raw/`. See `docs/DATASETS.md` for details.
