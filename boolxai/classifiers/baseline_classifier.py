# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

import logging
from copy import deepcopy
from typing import Callable, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import (
    _check_sample_weight,
    check_array,
    check_is_fitted,
    check_X_y,
)

from boolxai.rules.literal import Literal
from boolxai.rules.trivials import One, Zero
from boolxai.util import check_binary

logger = logging.getLogger(__name__)


class BaselineClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-style classifier for trivial or single-feature rules."""

    def __init__(
        self,
        metric: Callable = balanced_accuracy_score,
    ):
        """Instantiates a classifier based on the best trivial/single-feature rule.

        Args:
            metric (callable, optional): An sklearn-style metric function, see
                `sklearn.metrics`, used internally to score rules during training.
                Note: make sure to use the same metric for any external evaluation
                of the model, for example in cross-validation. Defaults to
                `balanced_accuracy_score`.
        """
        self.metric = metric

        self.best_score_ = None
        self.best_rule_ = None
        self.classes_ = None
        self.n_features_in_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "BaselineClassifier":
        """Finds the best simple rule exhaustively for training data X and labels y.

        Note: calling `fit()` populates the `best_rule_` and `best_score_` attributes.
        """
        # Check that X, y, and sample_weight have the correct shape
        X, y = check_X_y(X, y)
        num_rows, num_features = X.shape
        self.n_features_in_ = X.shape
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        # Check the validity of the labels and data
        check_binary(self.classes_)
        check_binary(np.unique(X))

        # Edge case - just one class present, return a trivial rule right away,
        # without evaluating all trivial and all single-feature rules
        if len(self.classes_) == 1:
            label = self.classes_[0]
            if label == 0:
                self.best_rule_ = Zero()
            elif label == 1:
                self.best_rule_ = One()
            else:
                raise ValueError("Single class but unidentified")
            self.best_score_ = BaselineClassifier._score_rule(
                self.best_rule_,
                X,
                y,
                self.metric,
                sample_weight,
                check=False,
            )
            return self

        # Evaluate the trivial rules
        # The initial rule is One
        self.best_rule_ = One()
        self.best_score_ = BaselineClassifier._score_rule(
            self.best_rule_,
            X,
            y,
            self.metric,
            sample_weight,
            check=False,
        )

        # We then check if Zero scores higher
        zero = Zero()
        zero_score = BaselineClassifier._score_rule(
            zero,
            X,
            y,
            self.metric,
            sample_weight,
            check=False,
        )
        if zero_score > self.best_score_:
            self.best_rule_ = zero
            self.best_score_ = zero_score
        logger.debug(
            f"Best trivial rule score is {self.best_score_=:.3f} for rule {self.best_rule_}"
        )

        # Now score all single-feature rules
        for literal_index in range(num_features):
            for negated in [True, False]:
                proposed_rule = Literal(literal_index, negated=negated)
                proposed_rule_score = BaselineClassifier._score_rule(
                    proposed_rule,
                    X,
                    y,
                    self.metric,
                    sample_weight,
                    check=False,
                )

                if proposed_rule_score > self.best_score_:
                    logger.debug(
                        f"""Updating single-feature best_score from {self.best_score_=:.3f} to 
                        {proposed_rule_score=:.3f} for rule {proposed_rule}"""
                    )
                    self.best_rule_ = deepcopy(proposed_rule)
                    self.best_score_ = proposed_rule_score

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates predictions for X with `self.best_rule_`.

        Args:
            X (np.ndarray): Input data - each row is a sample. Expected to be
                binary (i.e., 0/1/False/True).

        Returns:
            np.ndarray: A one-dimensional Boolean ndarray. Each value corresponds to
            the evaluation of the rule on the respective data row.
        """
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        return self.best_rule_.evaluate(X)

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """Generates predictions for X with `best_rule_` and returns their score vs. y.

        Note:
            The metric used to calculate the score is the one in the `metric` attribute.

        Args:
            X (np.ndarray): Input data - each row is a sample. Expected to be
                binary (i.e., 0/1/False/True).
            y (np.ndarray): A label for each sample in the input data. Expected
                to be binary (i.e., 0/1/False/True).
            sample_weight (np.ndarray, optional): A weight for each sample or
                None for equal sample weights. Defaults to None.

        Returns:
            float: The score of the rule in the `best_rule_` attribute, given the
            metric in the `metric` attribute.
        """
        # Input validation
        X, y = check_X_y(X, y)

        return BaselineClassifier._score_rule(
            self.best_rule_,
            X,
            y,
            self.metric,
            sample_weight,
            check=True,
        )

    @staticmethod
    def _score_rule(rule, X, y, metric, sample_weight, check):
        """Returns the score (using metric) of the given rule on X and y."""
        # Note the difference from score() - the latter returns the score of
        # best_rule, whereas this method returns the score of a given rule.

        # We allow the checking of inputs to be switched off here. The reason is
        # that in the context of a solver, it's enough to check the input just
        # once - re-checking each evaluation would be very wasteful.
        predictions = rule.evaluate(X, check)
        score = metric(y_true=y, y_pred=predictions, sample_weight=sample_weight)
        return score

    def _more_tags(self):
        return {"binary_only": True}
