# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

from typing import Tuple, Union

import numpy as np
import pytest

from boolxai import Literal, One, Operator, Wildcard, Zero
from boolxai.boolxai import BoolXAI

# Choose larger num_seeds for stronger tests, but this will also impact the time taken
num_seeds = 3
seeds = list(range(num_seeds))


def get_random_binary_array(seed: int, size: Union[int, Tuple[int]]) -> np.ndarray:
    """Returns a random binary array."""
    rng = np.random.default_rng(seed=seed)
    arr = rng.choice([0, 1], size=size)
    return arr


@pytest.mark.parametrize("invalid_element", [2, -1, "bla"])
def test_fit_invalid_label_raises(invalid_element):
    rule_classifier = BoolXAI.RuleClassifier()
    X = np.array([[0, 1]])
    y = np.array([invalid_element])
    with pytest.raises(ValueError):
        rule_classifier.fit(X, y)


@pytest.mark.parametrize("invalid_element", [2, -1, "bla"])
def test_raise_invalid_data_raises(invalid_element):
    rule_classifier = BoolXAI.RuleClassifier()
    X = np.array([[invalid_element, 1]])
    y = np.array([0])
    with pytest.raises(ValueError):
        rule_classifier.fit(X, y)


@pytest.mark.parametrize("y", [np.array([0, 1, 1, 1]), np.array([0, 0, 0, 1])])
@pytest.mark.parametrize("seed", seeds)
def test_fit_simple_problem(y, seed):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    rule_classifier = BoolXAI.RuleClassifier(
        num_starts=3, num_iterations=200, random_state=seed, num_jobs=1
    )
    rule_classifier.fit(X, y)
    y_pred = rule_classifier.predict(X)

    assert np.array_equal(y_pred, y)
    assert rule_classifier.best_score_ == 1.0


@pytest.mark.parametrize("y", [np.array([0, 0, 1, 1]), np.array([1, 0, 1, 0])])
@pytest.mark.parametrize("baseline", [True, False])
def test_fit_simple_problem_baseline(y, baseline):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    rule_classifier = BoolXAI.RuleClassifier(
        num_starts=1, num_iterations=0, random_state=42, num_jobs=1, baseline=baseline
    )
    rule_classifier.fit(X, y)

    if baseline:
        if np.array_equal(y, np.array([0, 0, 1, 1])):
            assert rule_classifier.best_rule_ == Literal(0)
        else:
            assert rule_classifier.best_rule_ == Literal(1, negated=True)
        assert rule_classifier.best_score_ == 1.0

    else:
        assert rule_classifier.best_rule_.complexity() == 3


@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_fit_reproducibility(seed, num_jobs):
    X = get_random_binary_array(seed, (20, 10))
    y = get_random_binary_array(seed, 20)

    # Same params for all classifiers, except for seed
    params = dict(num_starts=2, num_iterations=50, num_jobs=num_jobs)

    # First time
    rule_classifier = BoolXAI.RuleClassifier(random_state=seed, **params)
    rule_classifier.fit(X, y)

    # Second time (same seed)
    rule_classifier2 = BoolXAI.RuleClassifier(random_state=seed, **params)
    rule_classifier2.fit(X, y)

    # Third time (different seed)
    rule_classifier3 = BoolXAI.RuleClassifier(random_state=seed + 1, **params)
    rule_classifier3.fit(X, y)

    # Same seed should give the same score and rule
    assert rule_classifier.best_score_ == rule_classifier2.best_score_
    assert rule_classifier.best_rule_ == rule_classifier2.best_rule_

    assert rule_classifier.scores_ == rule_classifier2.scores_
    assert rule_classifier.rules_ == rule_classifier2.rules_

    # Different seed should give different score and rule
    assert rule_classifier.best_score_ != rule_classifier3.best_score_
    assert rule_classifier.best_rule_ != rule_classifier3.best_rule_

    assert rule_classifier.scores_ != rule_classifier3.scores_
    assert rule_classifier.rules_ != rule_classifier3.rules_


@pytest.mark.parametrize("seed", seeds)
def test_fit_max_samples(seed):
    X = get_random_binary_array(seed, (20, 10))
    y = get_random_binary_array(seed, 20)

    # Same params for all classifiers, except for max_samples
    params = dict(num_starts=1, num_iterations=100, random_state=seed, num_jobs=1)

    # First time - without max samples
    rule_classifier = BoolXAI.RuleClassifier(max_samples=None, **params)
    rule_classifier.fit(X, y)

    # Second time - with max_samples that is larger than num_samples
    rule_classifier2 = BoolXAI.RuleClassifier(max_samples=21, **params)
    rule_classifier2.fit(X, y)

    # Third time - with max_samples=2
    rule_classifier3 = BoolXAI.RuleClassifier(max_samples=2, **params)
    rule_classifier3.fit(X, y)

    # The first two should give the exact same results
    assert rule_classifier.scores_ == rule_classifier2.scores_
    assert rule_classifier.rules_ == rule_classifier2.rules_

    assert rule_classifier.best_score_ == rule_classifier2.best_score_
    assert rule_classifier.best_rule_ == rule_classifier2.best_rule_

    # We expect the latter to provide a different and in particular worse result
    assert rule_classifier.best_score_ > rule_classifier3.best_score_
    assert rule_classifier.best_rule_ != rule_classifier3.best_rule_

    assert rule_classifier.scores_ != rule_classifier3.scores_
    assert rule_classifier.rules_ != rule_classifier3.rules_


@pytest.mark.parametrize("seed", seeds)
def test_fit_sample_weight(seed):
    X = get_random_binary_array(seed, (20, 10))
    y = get_random_binary_array(seed, 20)

    # Same params for all classifiers, except for sample_weight
    params = dict(num_starts=1, num_iterations=100, random_state=seed, num_jobs=1)

    # First time - without sample_weight
    rule_classifier = BoolXAI.RuleClassifier(**params)
    rule_classifier.fit(X, y, sample_weight=None)

    # Second time - with sample_weight that is all 1's
    rule_classifier2 = BoolXAI.RuleClassifier(**params)
    rule_classifier2.fit(X, y, sample_weight=[1] * len(y))

    # Third time - with lop-sided sample_weight
    rule_classifier3 = BoolXAI.RuleClassifier(**params)
    rule_classifier3.fit(X, y, sample_weight=[0.01] * (len(y) - 3) + [0.99, 0.99, 0.99])

    # The first two should give the exact same results
    assert rule_classifier.scores_ == rule_classifier2.scores_
    assert rule_classifier.rules_ == rule_classifier2.rules_

    assert rule_classifier.best_score_ == rule_classifier2.best_score_
    assert rule_classifier.best_rule_ == rule_classifier2.best_rule_

    # We expect the latter to achieve a better result, since it is scored mostly on just
    # three samples, so it's easier to fit
    assert rule_classifier.best_score_ < rule_classifier3.best_score_
    assert rule_classifier.best_rule_ != rule_classifier3.best_rule_

    assert rule_classifier.scores_ != rule_classifier3.scores_
    assert rule_classifier.rules_ != rule_classifier3.rules_


@pytest.mark.parametrize("func, result", [[np.ones, One()], [np.zeros, Zero()]])
def test_fit_trivial(func, result):
    X = get_random_binary_array(seed=42, size=(20, 3))
    y = func(20)

    rule_classifier = BoolXAI.RuleClassifier()
    rule_classifier.fit(X, y)
    assert rule_classifier.best_rule_ == result
    assert rule_classifier.best_score_ == 1.0
    for rule, score in zip(rule_classifier.rules_, rule_classifier.scores_):
        assert rule == result
        assert score == 1.0


@pytest.mark.parametrize("seed", seeds)
def test_fit_cutoffs(seed):
    X = get_random_binary_array(seed, (10, 5))
    y = get_random_binary_array(seed, 10)

    rule_classifier = BoolXAI.RuleClassifier(
        num_starts=10,
        num_iterations=20,
        random_state=seed,
        num_jobs=1,
        max_complexity=3,
        max_depth=1,
        baseline=False,
    )
    rule_classifier.fit(X, y)

    assert rule_classifier.best_rule_.complexity() <= 3
    assert rule_classifier.best_rule_.depth() == 1

    for rule in rule_classifier.rules_:
        assert rule.complexity() <= 3
        assert rule.depth() == 1

    index = np.argmax(rule_classifier.scores_)
    assert rule_classifier.best_rule_ == rule_classifier.rules_[index]
    assert rule_classifier.best_score_ == rule_classifier.scores_[index]


@pytest.mark.parametrize("seed", seeds)
def test_predict_and_score(seed):
    X = get_random_binary_array(seed, (10, 5))
    y = get_random_binary_array(seed, 10)

    rule_classifier = BoolXAI.RuleClassifier(
        num_starts=1, num_iterations=10, random_state=seed, num_jobs=1
    )
    rule_classifier.fit(X, y)
    y_pred = rule_classifier.predict(X)
    score_from_metric = rule_classifier.metric(y_pred=y_pred, y_true=y)

    assert rule_classifier.score(X, y) == score_from_metric


@pytest.mark.parametrize(
    "base_rule",
    [
        Operator.Or([Literal(0), Wildcard(), Wildcard()]),  # Two wildcard nodes
        Operator.Or([Literal(0), Literal(1)]),  # No wildcard nodes
        Wildcard(),  # Wildcard at root
    ],
)
def test_fit_simple_problem_base_rule_raise(base_rule):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 1, 1])

    base_rule = Operator.Or([Literal(0), Wildcard(), Wildcard()])

    rule_classifier = BoolXAI.RuleClassifier(
        num_starts=1,
        num_iterations=0,
        random_state=42,
        num_jobs=1,
        baseline=False,
        base_rule=base_rule,
    )
    with pytest.raises(ValueError):
        rule_classifier.fit(X, y)


def test_fit_simple_problem_base_rule_typical():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 1, 1])

    base_rule = Operator.Or([Literal(0), Wildcard()])
    rule_classifier = BoolXAI.RuleClassifier(
        num_starts=2,
        num_iterations=0,
        random_state=42,
        num_jobs=1,
        baseline=False,
        base_rule=base_rule,
    )
    rule_classifier.fit(X, y)

    rules = [rule_classifier.best_rule_] + rule_classifier.rules_

    # No wildcards should appear
    for rule in rules:
        for node in rule.flatten():
            if isinstance(node, Wildcard):
                raise AssertionError("Found wildcard node")

    # Base rule should be present
    for rule in rules:
        assert isinstance(rule, Operator.Or)
        assert rule.subrules[0] == Literal(0)
        assert len(rule.subrules) == 2
