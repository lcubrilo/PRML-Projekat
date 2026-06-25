<!--
REPORT STRUCTURE STUB. Sections map 1:1 to the TA brief's required report contents
(introduction, problem, dataset, baselines, extension, setup, results, analysis,
conclusions) plus the brief's expected outputs (EDA, hyperparameter analysis,
visualizations, discussion).

Single file on purpose: the report must read as one document, and we are two authors.
Each section notes what goes in it and which notebook/issue/file feeds it.
Language: ENGLISH (decided).
Plots: saved to report/figures/ via `src.plotting.save_fig("name")`, referenced below as
figures/name.png. These image files may not exist yet; what each plot SHOWS gets written
once we see the actual plot.
Convert to PDF at the end (e.g. pandoc, which embeds the figures/*.png).
-->

# Comparative Study of Classical Classifiers and a Multiclass Boosting Extension for Predicting UFC Fight Outcomes (Winner and Method)

*Luka Čubrilo, Milica Cvetić. PMF UNS, Pattern Recognition and Machine Learning*

> Authoring split (by domain): **Milica** = §3 Dataset, §5.1 Baselines, §6 Experimental setup (factual / mechanical). **Luka** = §1 Intro, §2 Problem, §4 EDA, §8 Analysis, §9 Conclusions (domain / narrative). §5.2 Extension and §7 Results: shared.

## 1. Introduction
Motivation; the ~63–67% predictability ceiling and why a "low" number is still a finding; one-paragraph summary of what we do and the headline result. (See SOURCES.md.)

## 2. Problem description
The task: joint multiclass prediction of **winner × method** (6 classes). Why joint over cascade (TA's note + our reasoning). The "whose style wins" framing. Reference baselines we measure against (always-red, market odds).

## 3. Dataset description
mdabbert Ultimate UFC Dataset: source, size (~7k fights, 2010–2026), key columns, pre-fight aggregation, what we exclude (`total_fight_time_secs`, result columns) and why. Weird outcomes dropped (with reasoning). (See DATASETS.md.)

## 4. Exploratory data analysis  - feeds: `notebooks/01_eda.ipynb`
Class balance (6 classes + the 3-method view), weight-class/stance breakdowns, feature distributions, and the controlled **red-corner confound** analysis. Leakage sanity check (debut rows have empty priors).
Figures: `figures/class_balance.png`, `figures/method_distribution.png`, `figures/red_corner_confound.png`, `figures/feature_correlations.png`.

## 5. Methods
### 5.1 Baseline methods (from scratch) - feeds: `report/baselines_chosen.md`, `src/baselines/`
LDA, QDA, kNN: what they are, why these three, from-scratch + sklearn-validation note.
### 5.2 Extension (from scratch) - feeds: `src/extension/`
SAMME (multiclass AdaBoost): algorithm, decision stumps, reference (Zhu et al. 2009; Freund & Schapire 1997), why it suits a 6-class problem.

## 6. Experimental setup  - feeds: `src/data/`
Feature build (difference and/or absolute features; corner symmetrization), chronological 80/20 split, scaling fit on train only, validation-against-sklearn protocol, metrics, number of seeds, hyperparameter-sweep protocol.

## 7. Results  - feeds: `notebooks/02–04`
Baseline panel vs SAMME (mean ± std over seeds); confusion matrices; hyperparameter analysis (plots); odds benchmark (log-loss vs the market); optional PCA/LDA dimensionality-reduction ablation.
Figures: `figures/model_comparison.png`, `figures/confusion_matrix_samme.png`, `figures/hyperparam_k_knn.png`, `figures/hyperparam_samme.png`, `figures/pca_2d.png`.

## 8. Analysis and discussion
Does the extension beat the baselines, and is the gain worth the complexity? Does dimensionality reduction help? Feature importance / what predicts *how* a fight ends. How close did we get to the market and the ceiling. Honest discussion of errors and limitations (TA: errors are fine if explained).

## 9. Conclusions
What we found, what we would do next.

## References
SAMME: Zhu, Zou, Rosset & Hastie (2009). AdaBoost: Freund & Schapire (1997). Domain: ACM 2024 style-clustering paper; Kuhn, *Fightnomics*. Texts: Duda/Hart/Stork, Bishop, Hastie/Tibshirani/Friedman. (See SOURCES.md.)

## Appendix
Repo layout, how to run, the sklearn-validation results (`tests/`).
