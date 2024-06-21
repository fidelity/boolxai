# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

import logging
from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np

from boolxai.moves import _Move
from boolxai.rules import (
    Literal,
    One,
    Operator,
    ParametrizedOperator,
    UnparametrizedOperator,
    Zero,
    parametrized_operator_classes,
)

logger = logging.getLogger(__name__)


def get_random_indices(max_int: int, num_indices: int, seed: int) -> np.ndarray:
    """Returns num_indices indices up to max_int without replacement, using seed.

    Args:
        max_int (int): Choose indices up to this number (inclusive).
        num_indices (int): Number of indices to return.
        seed (int): Random number generator seed to use.

    Returns:
        np.ndarray: An array containing the randomly chosen indices.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(max_int, size=num_indices, replace=False)
    return indices


def generate_random_rule(
    literal_indices: Sequence[int],
    max_num_literals: int,
    operators: Sequence[
        Union[Type[ParametrizedOperator], Type[UnparametrizedOperator]]
    ],
    random_state: np.random.RandomState,
) -> Union[ParametrizedOperator, UnparametrizedOperator]:
    """Returns a random rule with up to max_num_literals from literal_indices.

    Args:
        literal_indices (list): Indices of the literals that can be included.
        max_num_literals (int): The maximal number of literals in the generated rule.
            Must be 2 or more.
        operators (list): List of operator classes from wich to choose the operator.
        random_state (np.random.RandomState): The random number generator that will be
            used.

    Returns:
        Union[ParametrizedOperator, UnparametrizedOperator]: The random rule.
    """
    if len(literal_indices) < 2:
        raise ValueError("Must pass in two or more literal_indices")

    actual_max_num_literals = min(max_num_literals, len(literal_indices))
    num_literals_to_choose = random_state.randint(
        2, actual_max_num_literals + 1
    )  # Note: numpy's randint() includes low but excludes high, hence plus one
    chosen_literal_indices = random_state.choice(
        literal_indices, num_literals_to_choose, replace=False
    )  # Samples without replacement
    literals = [
        Literal(literal_index, negated=random_state.choice([False, True]))
        for literal_index in chosen_literal_indices
    ]
    operator, param = operator_factory(operators, num_literals_to_choose, random_state)

    if param is None:
        rule = operator(literals)
    else:
        rule = operator(literals, param=param)

    return rule


def operator_factory(
    operators: Sequence[
        Union[Type[ParametrizedOperator], Type[UnparametrizedOperator]]
    ],
    num_subrules: int,
    random_state: np.random.RandomState,
) -> Tuple[
    Union[Type[ParametrizedOperator], Type[UnparametrizedOperator]], Optional[int]
]:
    """Returns a random operator class and parameter (only for parametrized operators).

    Note: for parameterized operators, the parameter is chosen randomly from the allowed
    values, i.e., such that 1 <= param <= num_subrules-1.

    Args:
        operators (list): The operator classes to include.
        num_subrules (int): The number of subrules under this operator. Used for
            choosing a valid parameter if the operator chosen is a parameterized
            operator.
        random_state (RandomState): The random number generator to use.

    Returns:
        A tuple containing a random operator class and valid parameter.
    """
    if num_subrules < 2:
        raise ValueError("Cannot be used with less than two subrules")

    if len(operators) == 1:
        operator_class = operators[0]
    else:
        operator_class = random_state.choice(operators)

    if operator_class in parametrized_operator_classes:
        max_param = num_subrules - 1
        allowed_param_values = range(1, max_param + 1)
        operator_param = random_state.choice(allowed_param_values)

    else:
        operator_param = None

    return operator_class, operator_param


def formulate_effective_problem(
    rule: Operator,
    target: Union[Literal, Operator],
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
):
    """Formulates effective problem for replacing target in rule.

    Given the original X and y, a rule and a target node in the rule, the effective
    problem is a new X' and y' (subsets of X, and y) that take into account the rule and
    target. For example, if base_rule = And(a,b) and a=0 in a particular row, then
    regardless of the value of b, the output will always be 0, so that row and label are
    dropped from the effective problem.

    Args:
        rule (Operator): The rule in which the replacement will take place.
        target (Literal or Operator): The subrule that will be replaced.
        X (np.ndarray): Input data - each row is a sample. Expected to be
            binary (i.e., 0/1/False/True).
        y (np.ndarray): A label for each sample in the input data. Expected
            to be binary (i.e., 0/1/False/True).
        sample_weight (np.ndarray, optional): A weight for each sample or
            None for equal sample weights. Defaults to None.

    Returns:
        A tuple containing, respectively, the effective training data, the effective
        labels, and the effective sample weights (all of type np.ndarray).
    """
    # To determine the effective problem, we copy the rule twice, replacing target with
    # One() and Zero() (respectively). Then we evaluate the resulting rules for all
    # inputs. We identify whether the output is the same - i.e., independent of the
    # value of target. We call those samples "pre-determined", and remove them from the
    # effective problem.
    y_one = _Move.find_and_replace(rule, target, One()).evaluate(X)
    y_zero = _Move.find_and_replace(rule, target, Zero()).evaluate(X)

    y_effective = []
    indices = []
    for i in range(len(y)):
        label = y[i]

        # Ignore the pre-determined case y_one[i]==y_zero[i], nothing to do
        if y_one[i] != y_zero[i]:
            if y_one[i] == label:
                y_effective.append(y_one[i])
            elif y_zero[i] == label:
                y_effective.append(y_zero[i])
            else:
                raise ValueError(
                    "y_one and y_zero are different but neither is equal to label"
                )
            indices.append(i)

    X_effective = X[indices, :]
    y_effective = np.array(y_effective)
    if sample_weight is not None:
        sample_weight_effective = sample_weight[indices, :]
    else:
        sample_weight_effective = None

    return X_effective, y_effective, sample_weight_effective
