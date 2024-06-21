# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

from typing import Tuple, Union

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from boolxai import BoolXAI, Literal, One, Zero


def get_random_binary_array(seed: int, size: Union[int, Tuple[int]]) -> np.ndarray:
    """Returns a random binary array."""
    rng = np.random.default_rng(seed=seed)
    arr = rng.choice([0, 1], size=size)
    return arr


@pytest.mark.parametrize("invalid_element", [2, -1, "bla"])
def test_fit_invalid_label_raises(invalid_element):
    baseline_classifier = BoolXAI.BaselineClassifier()
    X = np.array([[0, 1]])
    y = np.array([invalid_element])
    with pytest.raises(ValueError):
        baseline_classifier.fit(X, y)


@pytest.mark.parametrize("invalid_element", [2, -1, "bla"])
def test_raise_invalid_data_raises(invalid_element):
    baseline_classifier = BoolXAI.BaselineClassifier()
    X = np.array([[invalid_element, 1]])
    y = np.array([0])
    with pytest.raises(ValueError):
        baseline_classifier.fit(X, y)


@pytest.mark.parametrize("func, result", [[np.ones, One()], [np.zeros, Zero()]])
def test_fit_trivial(func, result):
    # If all labels are from the same class, should short-circuit
    X = get_random_binary_array(seed=42, size=(4, 3))
    y = func(4)

    baseline_classifier = BoolXAI.BaselineClassifier()
    baseline_classifier.fit(X, y)

    assert baseline_classifier.best_rule_ == result
    assert baseline_classifier.best_score_ == 1.0


@pytest.mark.parametrize(
    "y, result", [[np.array([1, 1, 0, 1]), One()], [np.array([1, 0, 0, 0]), Zero()]]
)
def test_fit_trivial2(y, result):
    # If there's a majority for a class, and the features all have a score of 0.5,
    # should return the trivial rule corresponding to the majority class - when using
    # the accuracy metric.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    baseline_classifier = BoolXAI.BaselineClassifier(metric=accuracy_score)
    baseline_classifier.fit(X, y)

    assert baseline_classifier.best_rule_ == result
    assert baseline_classifier.best_score_ == 0.75


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("negated", [False, True])
def test_fit_feature(index, negated):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    if negated:
        y = 1 - X[:, index]
    else:
        y = X[:, index]

    baseline_classifier = BoolXAI.BaselineClassifier()
    baseline_classifier.fit(X, y)

    assert baseline_classifier.best_rule_ == Literal(index, negated=negated)
    assert baseline_classifier.best_score_ == 1.0
