"""From-scratch baseline implementations, ported from the course materials
(`Project context/course_materials/`).

These are the *primary* implementations for the project - external libraries
(sklearn, etc.) are used ONLY to validate these (see `tests/`), never as the
source of a reported result.

Coverage of the covered course topics (all validated vs sklearn where one exists):
    02 linear_regression  -> linear.LinearRegression (normal / GD / ridge)
    03 least_squares      -> linear_classifiers (Perceptron, Ho-Kashyap, LeastSquares)
    04 bayes_decision     -> gaussian.GaussianClassifier(kind='nb')
    05 quadratic_class.   -> gaussian.GaussianClassifier(kind='qda' | 'lda')
    06 mle                -> the *fitting principle* used inside gaussian/linear
                             (means/covariances/weights are MLE estimates); not a
                             standalone model.
    07 kde                -> kde (KDE estimator + ParzenClassifier); course bugs fixed
    08 knn                -> knn.KNNClassifier
    09 pca                -> pca.PCA
    10 lda                -> gaussian (LDA *classifier*) + lda_projection (LDA as DR)
    11 clustering         -> kmeans.KMeans
"""
from .knn import KNNClassifier, knn_predict, euclidean_distance, manhattan_distance
from .gaussian import GaussianClassifier
from .pca import PCA
from .kmeans import KMeans, kmeans, kmeans_objective
from .linear import LinearRegression, polynomial_features
from .linear_classifiers import PerceptronClassifier, HoKashyapClassifier, LeastSquaresClassifier
from .kde import ParzenClassifier, kde_estimate, silverman_bandwidth
from .lda_projection import LDAProjection

__all__ = [
    "KNNClassifier", "knn_predict", "euclidean_distance", "manhattan_distance",
    "GaussianClassifier",
    "PCA",
    "KMeans", "kmeans", "kmeans_objective",
    "LinearRegression", "polynomial_features",
    "PerceptronClassifier", "HoKashyapClassifier", "LeastSquaresClassifier",
    "ParzenClassifier", "kde_estimate", "silverman_bandwidth",
    "LDAProjection",
]
