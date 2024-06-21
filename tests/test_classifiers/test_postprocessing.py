# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

from copy import deepcopy

import pytest

from boolxai import Literal, One, Operator, Wildcard, Zero
from boolxai.classifiers.postprocessing import remove_nested_and, remove_nested_or


@pytest.mark.parametrize("func", [remove_nested_and, remove_nested_or])
@pytest.mark.parametrize("rule", [Literal(0), One(), Zero(), Wildcard()])
def test_remove_nested_operator_nothing_to_do(func, rule):
    rule_backup = deepcopy(rule)
    func(rule)
    assert rule == rule_backup


def test_remove_nested_and_typical():
    rule = Operator.And(
        [Operator.And([Literal(0), Literal(1)]), Operator.And([Literal(1), Literal(2)])]
    )
    remove_nested_and(rule)
    assert rule == Operator.And([Literal(0), Literal(1), Literal(2)])


def test_remove_nested_or_typical():
    rule = Operator.Or(
        [
            Operator.Or([Literal(0), Literal(1)]),
            Operator.AtLeast([Literal(1), Literal(2)], param=1),
        ]
    )
    remove_nested_or(rule)
    assert rule == Operator.Or([Literal(0), Literal(1), Literal(2)])


def test_remove_nested_or_with_and():
    # This test failed in a previous implementation - nothing to do for Or(And(Or(..)))
    rule = Operator.Or(
        [Operator.And([Operator.Or([Literal(0), Literal(1)]), Literal(1)]), Literal(0)]
    )
    rule_copy = deepcopy(rule)
    remove_nested_or(rule)
    assert rule == rule_copy


def test_remove_nested_and_with_or():
    # This test failed in a previous implementation - nothing to do for And(Or(And(..)))
    rule = Operator.And(
        [Operator.Or([Operator.And([Literal(0), Literal(1)]), Literal(1)]), Literal(0)]
    )
    rule_copy = deepcopy(rule)
    remove_nested_and(rule)
    assert rule == rule_copy
