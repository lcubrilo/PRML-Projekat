"""Integration runner: (1) executes every per-method unit test in tests/unit/,
then (2) runs genuine cross-method integration checks (methods composed into a
pipeline). sklearn appears only for correctness comparison, never in results.

Run: python tests/validate_baselines.py
"""
import sys, os, importlib.util, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

results = []
def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")

# --- (1) run all unit tests by importing each module and calling its test_* fns ---
print("== unit tests ==")
for path in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "unit", "test_*.py"))):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fname in [n for n in dir(mod) if n.startswith("test_")]:
        try:
            getattr(mod, fname)()
            check(f"{os.path.basename(path)}::{fname}", True)
        except AssertionError as e:
            check(f"{os.path.basename(path)}::{fname}", False, str(e))

# --- (2) integration: methods composed together (what unit tests don't cover) ---
print("== integration (composed pipelines) ==")
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.baselines import PCA, GaussianClassifier, KNNClassifier, LDAProjection

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# scaling fit on TRAIN ONLY (the leakage-safe preprocessing we'll use on UFC data)
mu, sd = Xtr.mean(0), Xtr.std(0)
Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd
check("train-only scaling + kNN", KNNClassifier(k=5).fit(Xtr_s, ytr).score(Xte_s, yte) > 0.9)

# PCA (fit on train) -> Gaussian classifier on reduced features (the 'does PCA help?' branch)
p = PCA(n_components=2).fit(Xtr)
acc_pca = GaussianClassifier("qda").fit(p.transform(Xtr), ytr).score(p.transform(Xte), yte)
check("PCA(2) -> QDA pipeline", acc_pca > 0.9, f"acc={acc_pca:.3f}")

# LDA projection (fit on train) -> kNN
l = LDAProjection(n_components=2).fit(Xtr, ytr)
acc_lda = KNNClassifier(k=5).fit(l.transform(Xtr), ytr).score(l.transform(Xte), yte)
check("LDAProjection(2) -> kNN pipeline", acc_lda > 0.9, f"acc={acc_lda:.3f}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
