# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Portions copyright 2023 FMR LLC
# Portions copyright 2023 Amazon Web Services, Inc.

import pytest

from boolxai import Literal, One, Operator, Zero
from boolxai.moves import NoParentError, NotEnoughSubrulesError, _Move


@pytest.fixture
def a():
    return Literal(0)


@pytest.fixture
def not_a():
    return Literal(0, negated=True)


@pytest.fixture
def b():
    return Literal(1)


@pytest.fixture
def c():
    return Literal(2)


def test_remove_literal_typical(not_a, b, c):
    f = Operator.Or([not_a, b, c])
    not_a = f.subrules[0]
    b = f.subrules[1]
    _Move.remove_literal(not_a)

    assert f == Operator.Or([b, c])


def test_remove_literal_operator(not_a, b, c):
    f = Operator.Or([Operator.And([not_a, b]), c])
    and_operator = f.subrules[0]

    with pytest.raises(TypeError):
        _Move.remove_literal(and_operator)


def test_remove_literal_no_parent(a):
    with pytest.raises(NoParentError):
        _Move.remove_literal(a)


def test_remove_literal_not_enough_subrules_shallow(a, b, c):
    f = Operator.And([Operator.Or([b, a]), c])

    c = f.subrules[1]
    with pytest.raises(NotEnoughSubrulesError):
        _Move.remove_literal(c)


def test_remove_literal_not_enough_subrules_deep(a, b, c):
    f = Operator.And([Operator.Or([b, a]), c])

    b = f.subrules[0].subrules[0]
    with pytest.raises(NotEnoughSubrulesError):
        _Move.remove_literal(b)


@pytest.mark.parametrize(
    "operator_class, operator_param",
    [
        (Operator.And, None),
        (Operator.Or, None),
        (Operator.AtMost, 2),
        (Operator.AtLeast, 2),
        (Operator.Choose, 2),
    ],
)
def test_remove_literal_adjust_param(operator_class, operator_param, a, b, c):
    if operator_param is None:
        f = operator_class([a, b, c])
    else:
        f = operator_class([a, b, c], param=operator_param)
    b = f.subrules[1]
    _Move.remove_literal(b)

    if operator_param is not None:
        assert f.param == 1


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("negated", [True, False])
def test_swap_literal_typical(index, negated, b):
    _Move.swap_literal(b, index, negated)

    assert b.index == index
    assert b.negated == negated


@pytest.mark.parametrize(
    "operator_class, operator_param",
    [
        (Operator.And, None),
        (Operator.Or, None),
        (Operator.AtMost, 1),
        (Operator.AtLeast, 1),
        (Operator.Choose, 1),
    ],
)
def test_expand_literal_to_operator_typical(operator_class, operator_param, a, b, c):
    f = Operator.And([a, b, c])

    # b is expanded with the sibling c
    b = f.subrules[1]
    c = f.subrules[2]

    _Move.expand_literal_to_operator(b, c, operator_class, operator_param)
    if operator_param is None:
        g = operator_class([b, c])
    else:
        g = operator_class([b, c], param=operator_param)
    assert f == Operator.And([a, g])
    for subrule in f.subrules:
        assert subrule.parent == f


@pytest.mark.parametrize(
    "operator_class, operator_param",
    [
        (Operator.And, None),
        (Operator.Or, None),
        (Operator.AtMost, 1),
        (Operator.AtLeast, 1),
        (Operator.Choose, 1),
    ],
)
def test_expand_literal_to_operator_not_enough_subrules(
    operator_class, operator_param, a, b
):
    f = Operator.And([a, b])
    b = f.subrules[1]
    a = f.subrules[0]
    with pytest.raises(NotEnoughSubrulesError):
        _Move.expand_literal_to_operator(b, a, operator_class, operator_param)


@pytest.mark.parametrize(
    "operator_class, operator_param",
    [
        (Operator.And, None),
        (Operator.Or, None),
        (Operator.AtMost, 2),
        (Operator.AtLeast, 2),
        (Operator.Choose, 2),
    ],
)
def test_expand_literal_to_operator_adjust_param(
    operator_class, operator_param, a, b, c
):
    if operator_param is None:
        f = operator_class([a, b, c])
    else:
        f = operator_class([a, b, c], param=operator_param)
    b = f.subrules[1]
    a = f.subrules[0]
    _Move.expand_literal_to_operator(b, a, Operator.And)

    if operator_param is not None:
        assert f.param == 1


def test_remove_operator_no_parent(a, b):
    f = Operator.And([a, b])
    with pytest.raises(NoParentError):
        _Move.remove_operator(f)


def test_remove_operator_typical(not_a, b, c):
    operator = Operator.Or([not_a, b])
    f = Operator.And([operator, c])
    operator = f.subrules[0]
    _Move.remove_operator(operator)

    assert f == Operator.And([c, not_a, b])
    for subrule in f.subrules:
        assert subrule.parent == f


def test_remove_operator_redundant_literal(a, not_a, b, c):
    operator = Operator.Or([not_a, b, c])
    f = Operator.And([operator, a, b])
    operator = f.subrules[0]
    _Move.remove_operator(operator)

    assert f == Operator.And([a, b, c])
    for subrule in f.subrules:
        assert subrule.parent == f


def test_add_literal_typical(a, b, c):
    f = Operator.And([a, c])
    _Move.add_literal(b, f)

    assert f == Operator.And([a, c, b])
    assert f.subrules[2].parent == f


@pytest.mark.parametrize(
    "new_operator_class, new_operator_param",
    [
        (Operator.And, None),
        (Operator.Or, None),
        (Operator.AtMost, 1),
        (Operator.AtLeast, 1),
        (Operator.Choose, 1),
    ],
)
def test_swap_operator(new_operator_class, new_operator_param, a, b):
    f = Operator.And([b, a])
    _Move.swap_operator(f, new_operator_class, new_operator_param)

    assert isinstance(f, new_operator_class)
    assert f.subrules == [b, a]
    if new_operator_param is not None:
        assert f.param == new_operator_param


def test_swap_parametrized_to_unparameterized(a, b):
    f = Operator.AtLeast([b, a], param=1)
    assert hasattr(f, "param")
    _Move.swap_operator(f, Operator.Or)
    assert not hasattr(f, "param")


def test_replace_subrule_with_rule(not_a, b, c):
    f = Operator.Or([Operator.And([not_a, b]), c])
    subrule = f.subrules[0]
    rule = Operator.And([b, c], negated=True)
    _Move.replace(subrule, rule)
    assert f == Operator.Or([Operator.And([b, c], negated=True), c])

    assert f.subrules[0].subrules[0].parent == f.subrules[0]
    assert f.subrules[0].subrules[1].parent == f.subrules[0]


def test_replace_subrule_with_literal(not_a, b, c):
    f = Operator.Or([Operator.And([not_a, b]), c])
    subrule = f.subrules[0]
    rule = b
    _Move.replace(subrule, rule)
    assert f == Operator.Or([b, c])


def test_replace_subrule_with_trivial(not_a, b, c):
    f = Operator.Or([Operator.And([not_a, b]), c])
    subrule = f.subrules[0]
    rule = Zero()
    _Move.replace(subrule, rule)
    assert f == Operator.Or([Zero(), c])


def test_replace_rule_with_new_rule(not_a, b, c):
    f = Operator.Or([Operator.And([not_a, b]), c])
    new_rule = Operator.And([b, c])
    _Move.replace(f, new_rule)

    assert f == Operator.And([b, c])
    assert f.subrules[0].parent == f


def test_replace_rule_with_literal(not_a, b, c):
    f = Operator.Or([Operator.And([not_a, b]), c])
    new_rule = c
    _Move.replace(f, new_rule)

    assert f == c


def test_replace_rule_with_trivial(not_a, b, c):
    f = Operator.Or([Operator.And([not_a, b]), c])
    new_rule = One()
    _Move.replace(f, new_rule)

    assert f == One()


def test_find_and_replace_literal(not_a, a, b, c):
    f = Operator.Or([not_a, b, c])
    target = Literal(1)  # b
    new_target = Literal(0)  # a
    new_f = _Move.find_and_replace(f, target, new_target)
    assert new_f == Operator.Or([not_a, a, c])


def test_find_and_replace_raise(b, c):
    # b appears in two places, so cannot find and replace since we want to replace just
    # one of the occurrences, but don't know which one.
    f = Operator.Or([b, Operator.And([b, c])])
    target = Literal(1)  # b
    new_target = Literal(0)  # a
    with pytest.raises(ValueError):
        _Move.find_and_replace(f, target, new_target)
