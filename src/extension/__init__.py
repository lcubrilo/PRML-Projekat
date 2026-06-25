"""From-scratch project extension: SAMME multi-class AdaBoost
(Zhu, Zou, Rosset & Hastie, 2009). Primary implementation - sklearn appears
only in tests for validation, never as the source of a reported result.
"""
from .samme import SAMMEClassifier, DecisionStump

__all__ = ["SAMMEClassifier", "DecisionStump"]
