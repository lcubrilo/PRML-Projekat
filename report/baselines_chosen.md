# Baseline methods: which we chose and why

> Standalone justification to fold into the final report. The TA approved using 1 to 3 baselines (not all), with an explanation of the choice. We use **three: LDA, QDA, kNN**. Language note: written in English as a working draft; translate to Serbian for the submitted report if that is the chosen language.

## Context
The task is a single multiclass classification: **6 classes = winner (red / blue) × method (KO/TKO, submission, decision)**. Features are pre-fight, tabular (differences and/or absolute per-fighter values). Every baseline is implemented from scratch and validated against scikit-learn (see `tests/`).

## The chosen set: LDA, QDA, kNN
We pick these three because together they span the three classical decision-boundary families from the course, which gives a controlled, low-redundancy comparison:

- **LDA** (linear discriminant): linear boundaries, one shared covariance for all classes. The low-variance, higher-bias generative reference, the "linear floor."
- **QDA** (quadratic discriminant): per-class covariance, so curved boundaries. More flexible generative model.
- **kNN** (k nearest neighbours): non-parametric, instance-based, makes no distributional assumption; captures local structure.

So the panel is *linear vs quadratic vs instance-based*, the standard contrast that lets us discuss bias/variance and boundary complexity without three methods that all do the same thing.

Why each, specifically:
- **QDA is the theoretically motivated centerpiece.** The six outcome classes plausibly differ in *covariance structure*, not just in means: a KO-prone matchup and a decision-prone matchup spread differently across striking/grappling stats. QDA can model per-class covariance; LDA cannot. Putting them side by side directly tests whether that extra flexibility pays off.
- **LDA is the cheap, informative reference.** If LDA is about as good as QDA, the classes effectively share covariance and the simpler model should be preferred (Occam). If QDA clearly wins, that itself is a finding about the data.
- **kNN is the non-parametric counterpoint.** If the class regions are non-Gaussian or multimodal, kNN can win where the Gaussian models fail. It is also a clean vehicle for two course themes: feature scaling (it is distance-based, so we scale on train only) and a hyperparameter sweep over k.

All three are **natively multiclass**, so no one-vs-rest workarounds are needed on the 6-class target.

## Why not the other covered methods
- **Perceptron / least-squares classifier:** inherently *binary linear*; would need one-vs-rest for 6 classes and add little beyond LDA as a linear model. Kept available as a linear floor but not central.
- **Gaussian Naive Bayes:** essentially QDA with a diagonal covariance (feature-independence assumption). Largely redundant once we already have LDA and QDA spanning shared and full covariance. Easy swap-in if we want the independence angle.
- **KDE / Parzen classifier:** density-based and conceptually close to kNN; redundant as a second non-parametric method.
- **k-means, PCA, linear regression:** wrong task (clustering / dimensionality reduction / regression), not classifiers. PCA still appears, but separately, as an optional dimensionality-reduction ablation, not as a baseline classifier.

## How this sets up the extension
The extension, **SAMME** (multiclass AdaBoost), is a fourth and deliberately different inductive bias: an additive ensemble of weak learners (decision stumps). The full comparison therefore spans generative-linear (LDA), generative-quadratic (QDA), instance-based (kNN), and boosting (SAMME). That is exactly the setup for the project's core research question: *does the more complex ensemble beat the classical baselines, and is the accuracy gain worth the added complexity?*

## Leaner variant, if wanted
If a smaller panel is preferred, **QDA + kNN** alone still gives the quadratic-vs-instance-based contrast. We keep LDA because the LDA-vs-QDA comparison is itself cheap and informative (it answers the shared-covariance question).
