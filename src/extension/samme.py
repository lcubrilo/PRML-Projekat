"""SAMME multi-class AdaBoost - from scratch (Zhu, Zou, Rosset & Hastie, 2009).

Identical to binary AdaBoost except the stump weight alpha gains a `+ log(K-1)`
term, which lets a weak learner that merely beats random guessing (1/K) still
contribute positively. With K=2 that term is 0 and this reduces exactly to
AdaBoost - which is why the binary case is the first validation checkpoint.

This is the *primary* implementation; sklearn appears only in tests/ for
validation, never as the source of a reported number. API mirrors the baselines
in src/baselines (fit/predict/predict_proba/score) and handles arbitrary
(e.g. string) labels by encoding them to 0..K-1 internally.
"""
from __future__ import annotations
import numpy as np


class DecisionStump:
    """A depth-1 decision tree (aka stump; one feature, one threshold) fit on *weighted*
    samples. This is the weak learner SAMME boosts - the meatiest part to write.

    Attributes to set in `fit` (suggested):
        feature_index : int   - column the split tests
        threshold     : float - split point; x[feature] <= threshold goes left
        left_class    : int   - encoded class predicted on the left side
        right_class   : int   - encoded class predicted on the right side
    """

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> "DecisionStump":
        """Pick the (feature, threshold) split that minimises weighted error.

        For every feature, scan candidate thresholds (midpoints between sorted
        unique values). Each split sends rows with x[feature] <= threshold left,
        the rest right; each side predicts its weighted-majority class. Score the
        split by the weighted misclassification `sum(sample_weight[i] for the
        rows it gets wrong)`. Keep the lowest-scoring split.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features), float
        y : np.ndarray, shape (n_samples,), encoded labels 0..K-1
        sample_weight : np.ndarray, shape (n_samples,), sums to 1 - boosting's
            ONLY channel for telling the stump which rows matter this round.

        Returns
        -------
        self
        """
        n_samples, n_features = X.shape
        K = int(y.max()) + 1
        best_error = np.inf

        for feature in range(n_features):
            col = X[:, feature]
            uniq = np.unique(col)
            if uniq.size < 2: # how to find threshold if we only have 1 unique value?
                continue # try a different column

            # continuous columns have thousands of unique values -> thousands of scans per
            # feature. cap to ~64 quantile cut points: tiny accuracy cost, ~100x faster.
            if uniq.size > 64:
                uniq = np.unique(np.quantile(uniq, np.linspace(0, 1, 65)))

            thresholds = (uniq[:-1] + uniq[1:]) / 2.0 # all possible thresholds i.e between any unique values

            for threshold in thresholds:
                left_mask = col <= threshold # gives us which rows in our column are below threshold

                # we count how many instances of each class we found for those left of threshold and those right of it
                # but weighted by importance
                # we say minlength=K to allow 0s for unrepresented classes, so no shifting of indeces happens
                left_w = np.bincount(y[left_mask], weights=sample_weight[left_mask], minlength=K)
                right_w = np.bincount(y[~left_mask], weights=sample_weight[~left_mask], minlength=K)

                left_class, right_class = int(np.argmax(left_w)), int(np.argmax(right_w)) # find left's and right's most prominent class

                pred = np.where(left_mask, left_class, right_class) # just two predictions, (2 most prominents)
                error = sample_weight[pred != y].sum() # count up how many misses we had, but weighted

                # we find best feature/threshold pair for current dataset+weights
                if error < best_error:
                    best_error = error
                    self.feature_index, self.threshold = feature, float(threshold)
                    self.left_class, self.right_class = left_class, right_class

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Apply the stored split.

        Returns
        -------
        np.ndarray, shape (n_samples,) - encoded labels: `left_class` where
        x[feature_index] <= threshold, else `right_class`.
        """
        # just apply the best one found in fit
        left_mask = X[:, self.feature_index] <= self.threshold
        return np.where(left_mask, self.left_class, self.right_class)


class SAMMEClassifier:
    """Multi-class AdaBoost via the SAMME algorithm.

    Attributes to set in `fit` (suggested):
        classes_ : np.ndarray, shape (K,) - original labels; index = encoded id
        K_       : int - number of classes
        stumps_  : list[DecisionStump] - one weak learner per kept round
        alphas_  : list[float] - the vote weight of each stump
    """

    def __init__(self, n_estimators: int = 50) -> None:
        """Store hyperparameters only - no fitting happens here.

        Parameters
        ----------
        n_estimators : int - number of boosting rounds (weak learners).
        """
        self.n_estimators = n_estimators

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SAMMEClassifier":
        """Run the SAMME boosting loop.

        1. Encode labels -> self.classes_, y_enc; set self.K_ = len(self.classes_).
        2. Init weights  w = ones(n) / n.
        3. For each of n_estimators rounds:
             a. stump = DecisionStump().fit(X, y_enc, w)
             b. miss  = stump.predict(X) != y_enc
             c. err   = (w * miss).sum() / w.sum()
             d. alpha = log((1 - err) / err) + log(self.K_ - 1)   # <- the SAMME term
             e. w    *= exp(alpha * miss);  then renormalise  w /= w.sum()
             f. append stump and alpha to self.stumps_ / self.alphas_
           Guard the edges: err == 0 (perfect stump -> cap alpha / stop) and
           err >= (K-1)/K (worse than random -> skip/stop) to avoid log blow-ups.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features), float
        y : np.ndarray, shape (n_samples,) - arbitrary labels (encoded internally)

        Returns
        -------
        self
        """
        self.classes_, y_enc = np.unique(y, return_inverse=True) # sorted classes indexed (we use indeces later on)
        self.K_ = len(self.classes_)
        n_samples = X.shape[0]

        w = np.ones(n_samples) / n_samples # start with uniform probabilities, they'll be updated over time
        self.stumps_, self.alphas_ = [], []

        for _ in range(self.n_estimators):
            stump = DecisionStump().fit(X, y_enc, w) # primitive learner
            misses = stump.predict(X) != y_enc
            err = (w * misses).sum() / w.sum() # weighted and normalized error 

            # this is the generalization over adaboost - not looking if bigger than 1/2, but if bigger than k-1/k
            if err >= (self.K_ - 1) / self.K_: 
                break # this stump is literally worse than random chance

            if err == 0: # if perfect
                self.stumps_.append(stump)
                self.alphas_.append(np.log((1 - 1e-10) / 1e-10) + np.log(self.K_ - 1)) # to avoid div by 0, use absurdly small number
                break # can't be better than this
            
            # last added term is also generalization over adaboost
            alpha = np.log((1-err)/err) + np.log(self.K_-1) 
            # alpha says how important this stump is - less error it has, more important it is and viceversa
            
            w = w * np.exp(alpha * misses) # give more weight for next iteration to those who were wrong the last time
            w = w / w.sum() # normalize weights

            self.stumps_.append(stump)
            self.alphas_.append(alpha)
        
        return self

    def _decision_scores(self, X: np.ndarray) -> np.ndarray:
        """Aggregate the ensemble's votes - shared by predict and predict_proba,
        so implement it once.

        Returns
        -------
        np.ndarray, shape (n_samples, K) - entry [i, k] is the summed alpha of
        the stumps that voted class k for sample i.
        """
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self.K_))
        for stump, alpha in zip(self.stumps_, self.alphas_):
            pred = stump.predict(X)
            scores[np.arange(n_samples), pred] += alpha

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Argmax over the decision scores, mapped back to original labels.

        Returns
        -------
        np.ndarray, shape (n_samples,) - values drawn from self.classes_.
        """
        return self.classes_[np.argmax(self._decision_scores(X), axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Normalise the decision scores so each row sums to 1.

        Needed by the E1 metrics (log-loss / ROC-AUC). Not calibrated
        probabilities - just the relative vote mass per class.

        Returns
        -------
        np.ndarray, shape (n_samples, K) - rows non-negative and sum to 1.
        """
        scores = self._decision_scores(X)
        return scores / scores.sum(axis=1, keepdims=True)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean accuracy of predict(X) against y."""
        return (self.predict(X) == y).mean()

    def staged_predict(self, X: np.ndarray):
        """Yield the ensemble's prediction after each successive boosting round.

        Accumulates the vote scores incrementally (one stump at a time) so you
        get the prediction for an ensemble of size 1, 2, ... T without refitting.
        Used to plot accuracy vs n_estimators for the E2 hyperparameter analysis.

        Yields
        ------
        np.ndarray, shape (n_samples,) - the prediction after t stumps, t=1..T.
        """
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self.K_))
        for stump, alpha in zip(self.stumps_, self.alphas_):
            scores[np.arange(n_samples), stump.predict(X)] += alpha
            yield self.classes_[np.argmax(scores, axis=1)]

    def staged_score(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Accuracy after each boosting round - the convergence curve for E2.

        Returns
        -------
        list[float], length len(self.stumps_) - accuracy of the size-t ensemble.
        """
        return [float((pred == y).mean()) for pred in self.staged_predict(X)]
