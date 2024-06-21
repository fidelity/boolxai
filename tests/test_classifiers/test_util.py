# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

import numpy as np
import pytest

from boolxai import (
    Literal,
    Operator,
    all_operator_classes,
    parametrized_operator_classes,
)
from boolxai.classifiers.util import (
    formulate_effective_problem,
    generate_random_rule,
    get_random_indices,
    operator_factory,
)


@pytest.fixture
def X():
    """Input for an AND function"""
    return np.array([[0, 0], [0, 1], [1, 0], [1, 1]])


@pytest.fixture
def y():
    """Labels for an AND function"""
    return np.array([0, 0, 0, 1])


# Choose larger num_seeds for stronger tests, but this will also impact the time taken
num_seeds = 3
seeds = list(range(num_seeds))


@pytest.mark.parametrize("seed", seeds)
def test_get_random_indices(seed):
    indices = get_random_indices(10, 8, seed)
    assert len(indices) == 8
    assert all(0 <= index <= 10 for index in indices)
    assert len(set(indices)) == len(indices)


@pytest.mark.parametrize("seed", seeds)
def test_get_random_indices_reproducibility(seed):
    indices = get_random_indices(500, 10, seed=seed)
    indices2 = get_random_indices(500, 10, seed=seed)
    indices3 = get_random_indices(500, 10, seed=seed + 1)

    # Same seed should give the same indices
    assert np.array_equal(indices, indices2)

    # Different seed should give different indices
    assert not np.array_equal(indices, indices3)
    assert not np.array_equal(indices2, indices3)


@pytest.mark.parametrize("seed", seeds)
def test_generate_random_rule(seed):
    literal_indices = [1, 21, 123, 10, 11, 12]
    max_num_literals = 4
    operators = all_operator_classes
    rule = generate_random_rule(
        literal_indices,
        max_num_literals,
        operators,
        random_state=np.random.RandomState(seed),
    )
    assert len(rule.subrules) <= max_num_literals
    assert type(rule) in operators


@pytest.mark.parametrize("seed", seeds)
def test_generate_random_rule_reproducibility(seed):
    literal_indices = [1, 21, 123, 10, 11, 12]
    max_num_literals = 4
    operators = all_operator_classes
    rule = generate_random_rule(
        literal_indices,
        max_num_literals,
        operators,
        random_state=np.random.RandomState(seed),
    )
    rule2 = generate_random_rule(
        literal_indices,
        max_num_literals,
        operators,
        random_state=np.random.RandomState(seed),
    )
    rule3 = generate_random_rule(
        literal_indices,
        max_num_literals,
        operators,
        random_state=np.random.RandomState(seed + 1),
    )

    # Same seed should give the same indices
    assert np.array_equal(rule, rule2)

    # Different seed should give different indices
    assert not np.array_equal(rule, rule3)
    assert not np.array_equal(rule2, rule3)


@pytest.mark.parametrize("seed", seeds)
def test_operator_factory(seed):
    operator_class, param = operator_factory(
        operators=all_operator_classes,
        num_subrules=4,
        random_state=np.random.RandomState(seed),
    )
    assert operator_class in all_operator_classes
    if operator_class in parametrized_operator_classes:
        assert param is not None
        assert 1 <= param <= 3  # = 4 - 1
    else:
        assert param is None


@pytest.mark.parametrize("seed", seeds)
def test_operator_factory_reproducibility(seed):
    operator_class, param = operator_factory(
        operators=all_operator_classes,
        num_subrules=5,
        random_state=np.random.RandomState(seed),
    )
    operator_class2, param2 = operator_factory(
        operators=all_operator_classes,
        num_subrules=5,
        random_state=np.random.RandomState(seed),
    )

    assert operator_class2 == operator_class
    assert param2 == param


@pytest.mark.parametrize(
    "subrule_index,expected_X_effective, expected_y_effective",
    [
        (0, np.array([[0, 1, 1], [1, 1, 0]]), np.array([0, 0])),
        (1, np.array([[1, 0, 0], [1, 1, 0]]), np.array([1, 0])),
    ],
)
def test_formulate_effective_problem_typical_predetermined_and_or(
    subrule_index, expected_X_effective, expected_y_effective
):
    f0, f1, f2 = Literal(0), Literal(1), Literal(2)
    proposed_rule = Operator.And([f0, Operator.Or([f1, f2])])
    target = proposed_rule.subrules[subrule_index]

    X = np.array(
        [
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 1, 0],
        ]
    )
    y = np.array([1, 0, 1, 0])

    (X_effective, y_effective, sample_weight_effective) = formulate_effective_problem(
        proposed_rule, target, X, y
    )

    assert np.array_equal(X_effective, expected_X_effective)
    assert np.array_equal(y_effective, expected_y_effective)
    assert sample_weight_effective is None


@pytest.mark.parametrize(
    "subrule_index,expected_X_effective, expected_y_effective",
    [
        (0, np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]), np.array([1, 1, 0])),
        (1, np.array([[0, 0, 0], [0, 1, 1]]), np.array([1, 0])),
    ],
)
def test_formulate_effective_problem_typical_predetermined_or_and(
    subrule_index, expected_X_effective, expected_y_effective
):
    f0, f1, f2 = Literal(0), Literal(1), Literal(2)
    proposed_rule = Operator.Or([f0, Operator.And([f1, f2])])
    target = proposed_rule.subrules[subrule_index]

    X = np.array(
        [
            [0, 0, 0],  # target 0 gives 0, target 1 gives 1
            [0, 1, 1],  # target 0 gives 0, target 1 gives 1
            [1, 0, 0],  # pre-determined due to f0=1
            [1, 1, 0],  # pre-determined due to f0=1
        ]
    )
    y = np.array([1, 0, 1, 0])

    (X_effective, y_effective, sample_weight_effective) = formulate_effective_problem(
        proposed_rule, target, X, y
    )

    assert np.array_equal(X_effective, expected_X_effective)
    assert np.array_equal(y_effective, expected_y_effective)
    assert sample_weight_effective is None


def test_formulate_effective_problem_root(X, y):
    f0, f1 = Literal(0), Literal(1)
    proposed_rule = Operator.And([f0, f1])
    target = proposed_rule

    (X_effective, y_effective, sample_weight_effective) = formulate_effective_problem(
        proposed_rule, target, X, y
    )

    assert np.array_equal(X_effective, X)
    assert np.array_equal(y_effective, y)
    assert sample_weight_effective is None


def test_formulate_effective_problem_all_predetermined():
    f0, f1 = Literal(0), Literal(1)
    proposed_rule = Operator.Or([f0, f1])
    target = proposed_rule.subrules[0]

    X = np.array(
        [
            [0, 1],  # pre-determined due to f1=1
            [1, 1],  # pre-determined due to f0=f1=1
        ]
    )
    y = np.array([1, 0])

    (X_effective, y_effective, sample_weight_effective) = formulate_effective_problem(
        proposed_rule, target, X, y
    )

    assert np.array_equal(X_effective, np.empty((0, 2)))
    assert np.array_equal(y_effective, np.array([]))
    assert sample_weight_effective is None


def assert_effective_problem_valid(X_effective, y_effective, X, y):
    """Check that X_effective and y_effective are valid for X, y.

    Checks:
    1. That all rows in X_effective appear in X.
    2. That the labels match up.
    3. That no row in X appears twice in X_effective. With real data, this could
        actually occur due to duplicity in X, but we won't pass in data like that
        for out tests.
    """
    visited_rows = np.zeros(len(y))

    for new_row, new_label in zip(X_effective, y_effective):
        for i, (row, label) in enumerate(zip(X, y)):
            if np.array_equal(new_row, row):
                assert np.array_equal(new_label, label)
                assert visited_rows[i] == 0
                visited_rows[i] = 1
                break
        else:
            raise AssertionError("new_row not found in X_effective")
