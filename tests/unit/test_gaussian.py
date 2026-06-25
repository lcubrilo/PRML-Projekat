import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from src.baselines import GaussianClassifier

def _split():
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

def test_lda_qda_nb_match_sklearn():
    Xtr, Xte, ytr, yte = _split()
    for kind, skl in [("lda", LinearDiscriminantAnalysis()),
                      ("qda", QuadraticDiscriminantAnalysis()),
                      ("nb",  GaussianNB())]:
        ours = GaussianClassifier(kind=kind).fit(Xtr, ytr).predict(Xte)
        ref = skl.fit(Xtr, ytr).predict(Xte)
        assert np.mean(ours == ref) >= 0.93, kind


if __name__ == '__main__':
    g = dict(globals())
    fns = [(n, f) for n, f in g.items() if n.startswith('test_') and callable(f)]
    for n, f in fns:
        f(); print('PASS', n)
    print(f'{len(fns)} unit checks passed')
