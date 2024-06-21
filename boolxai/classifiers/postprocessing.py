# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

from boolxai import Operator
from boolxai.moves import _Move


def _is_or(rule):
    return isinstance(rule, Operator.Or) or (
        isinstance(rule, Operator.AtLeast) and rule.param == 1
    )


def _is_and(rule):
    return isinstance(rule, Operator.And)


def _remove_nested_operators(rule, checker):
    """Removes nested operators where parent and child have the same operator."""
    nested_operators = [
        subrule
        for subrule in rule.flatten()
        if checker(subrule) and checker(subrule.parent)
    ]
    for nested_operator in nested_operators:
        _Move.remove_operator(nested_operator)


def remove_nested_or(rule):
    """Removes nested Or operators (in place), like Or(a, Or(b,c)) -> Or(a,b,c).

    Args:
        rule (Rule): The rule to be postprocessed.
    """
    return _remove_nested_operators(rule, _is_or)


def remove_nested_and(rule):
    """Removes nested And operators (in place), like And(a, And(b,c)) -> And(a,b,c).

    Args:
        rule (Rule): The rule to be postprocessed.
    """
    return _remove_nested_operators(rule, _is_and)
